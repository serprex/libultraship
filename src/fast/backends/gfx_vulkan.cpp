
#ifdef ENABLE_VULKAN

#define VMA_IMPLEMENTATION
#include "fast/backends/gfx_vulkan.h"

#include "fast/interpreter.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"

#include <SDL2/SDL_vulkan.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_sdl2.h>

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <stdexcept>

// ---- helpers ----------------------------------------------------------------

#define VK_CHECK(expr)                                                         \
    do {                                                                       \
        VkResult _r = (expr);                                                  \
        if (_r != VK_SUCCESS) {                                                \
            SPDLOG_ERROR("Vulkan error {} at {}:{}", (int)_r, __FILE__, __LINE__); \
            abort();                                                            \
        }                                                                      \
    } while (0)

namespace Fast {

// ---- GLSL generation --------------------------------------------------------
// We generate Vulkan GLSL 4.50 directly in C++ rather than using the prism
// template system (which targets OpenGL). The attribute layout exactly mirrors
// what the OpenGL template produces so the interpreter's vertex data is
// compatible.

static const char* vk_shader_item_to_str(uint32_t item, bool withAlpha, bool onlyAlpha,
                                          bool inputsHaveAlpha, bool firstCycle,
                                          bool hintSingleElement) {
    if (!onlyAlpha) {
        switch (item) {
            case SHADER_0:    return withAlpha ? "vec4(0.0,0.0,0.0,0.0)" : "vec3(0.0,0.0,0.0)";
            case SHADER_1:    return withAlpha ? "vec4(1.0,1.0,1.0,1.0)" : "vec3(1.0,1.0,1.0)";
            case SHADER_INPUT_1: return withAlpha||!inputsHaveAlpha ? "vInput1":"vInput1.rgb";
            case SHADER_INPUT_2: return withAlpha||!inputsHaveAlpha ? "vInput2":"vInput2.rgb";
            case SHADER_INPUT_3: return withAlpha||!inputsHaveAlpha ? "vInput3":"vInput3.rgb";
            case SHADER_INPUT_4: return withAlpha||!inputsHaveAlpha ? "vInput4":"vInput4.rgb";
            case SHADER_TEXEL0:
                return firstCycle ? (withAlpha?"texVal0":"texVal0.rgb") : (withAlpha?"texVal1":"texVal1.rgb");
            case SHADER_TEXEL0A:
                if (hintSingleElement) return firstCycle ? "texVal0.a" : "texVal1.a";
                return firstCycle
                    ? (withAlpha?"vec4(texVal0.a,texVal0.a,texVal0.a,texVal0.a)":"vec3(texVal0.a,texVal0.a,texVal0.a)")
                    : (withAlpha?"vec4(texVal1.a,texVal1.a,texVal1.a,texVal1.a)":"vec3(texVal1.a,texVal1.a,texVal1.a)");
            case SHADER_TEXEL1A:
                if (hintSingleElement) return firstCycle ? "texVal1.a" : "texVal0.a";
                return firstCycle
                    ? (withAlpha?"vec4(texVal1.a,texVal1.a,texVal1.a,texVal1.a)":"vec3(texVal1.a,texVal1.a,texVal1.a)")
                    : (withAlpha?"vec4(texVal0.a,texVal0.a,texVal0.a,texVal0.a)":"vec3(texVal0.a,texVal0.a,texVal0.a)");
            case SHADER_TEXEL1:
                return firstCycle ? (withAlpha?"texVal1":"texVal1.rgb") : (withAlpha?"texVal0":"texVal0.rgb");
            case SHADER_COMBINED: return withAlpha ? "texel" : "texel.rgb";
            case SHADER_NOISE:
                return withAlpha
                    ? "vec4(random(vec3(floor(gl_FragCoord.xy*pc.noiseScale),float(pc.frameCount))),random(vec3(floor(gl_FragCoord.xy*pc.noiseScale)+vec2(1,0),float(pc.frameCount))),random(vec3(floor(gl_FragCoord.xy*pc.noiseScale)+vec2(0,1),float(pc.frameCount))),random(vec3(floor(gl_FragCoord.xy*pc.noiseScale)+vec2(1,1),float(pc.frameCount))))"
                    : "vec3(random(vec3(floor(gl_FragCoord.xy*pc.noiseScale),float(pc.frameCount))),random(vec3(floor(gl_FragCoord.xy*pc.noiseScale)+vec2(1,0),float(pc.frameCount))),random(vec3(floor(gl_FragCoord.xy*pc.noiseScale)+vec2(0,1),float(pc.frameCount))))";
        }
    } else {
        switch (item) {
            case SHADER_0:       return "0.0";
            case SHADER_1:       return "1.0";
            case SHADER_INPUT_1: return "vInput1.a";
            case SHADER_INPUT_2: return "vInput2.a";
            case SHADER_INPUT_3: return "vInput3.a";
            case SHADER_INPUT_4: return "vInput4.a";
            case SHADER_TEXEL0:  return firstCycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL0A: return firstCycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL1A: return firstCycle ? "texVal1.a" : "texVal0.a";
            case SHADER_TEXEL1:  return firstCycle ? "texVal1.a" : "texVal0.a";
            case SHADER_COMBINED: return "texel.a";
            case SHADER_NOISE:
                return "random(vec3(floor(gl_FragCoord.xy*pc.noiseScale),float(pc.frameCount)))";
        }
    }
    return "";
}

static void vk_append_formula(std::string& out, const int c[2][4],
                               bool doSingle, bool doMultiply, bool doMix,
                               bool withAlpha, bool onlyAlpha, bool optAlpha,
                               bool firstCycle) {
    if (doSingle) {
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][3], withAlpha, onlyAlpha, optAlpha, firstCycle, false);
    } else if (doMultiply) {
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][0], withAlpha, onlyAlpha, optAlpha, firstCycle, false);
        out += " * ";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][2], withAlpha, onlyAlpha, optAlpha, firstCycle, true);
    } else if (doMix) {
        out += "mix(";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][1], withAlpha, onlyAlpha, optAlpha, firstCycle, false);
        out += ", ";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][0], withAlpha, onlyAlpha, optAlpha, firstCycle, false);
        out += ", ";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][2], withAlpha, onlyAlpha, optAlpha, firstCycle, true);
        out += ")";
    } else {
        out += "(";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][0], withAlpha, onlyAlpha, optAlpha, firstCycle, false);
        out += " - ";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][1], withAlpha, onlyAlpha, optAlpha, firstCycle, false);
        out += ") * ";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][2], withAlpha, onlyAlpha, optAlpha, firstCycle, true);
        out += " + ";
        out += vk_shader_item_to_str(c[onlyAlpha ? 1 : 0][3], withAlpha, onlyAlpha, optAlpha, firstCycle, false);
    }
}

std::string GfxRenderingAPIVulkan::BuildVertexShader(const CCFeatures& f, uint8_t& numFloatsOut,
                                                      uint8_t& numAttribsOut, uint8_t attribSizes[16]) {
    std::string s;
    s += "#version 450\n";

    uint8_t loc = 0;
    uint8_t numFloats = 4; // position is always 4 floats
    uint8_t cnt = 0;

    // Position (always)
    s += "layout(location=0) in vec4 aVtxPos;\n";
    attribSizes[cnt++] = 4;

    for (int i = 0; i < 2; i++) {
        if (f.usedTextures[i]) {
            s += "layout(location=" + std::to_string(cnt) + ") in vec2 aTexCoord" + std::to_string(i) + ";\n";
            s += "layout(location=" + std::to_string(cnt) + ") out vec2 vTexCoord" + std::to_string(i) + ";\n";
            attribSizes[cnt++] = 2;
            numFloats += 2;

            for (int j = 0; j < 2; j++) {
                if (f.clamp[i][j]) {
                    std::string name = std::string(j == 0 ? "aTexClampS" : "aTexClampT") + std::to_string(i);
                    std::string vname = std::string(j == 0 ? "vTexClampS" : "vTexClampT") + std::to_string(i);
                    s += "layout(location=" + std::to_string(cnt) + ") in float " + name + ";\n";
                    s += "layout(location=" + std::to_string(cnt) + ") out float " + vname + ";\n";
                    attribSizes[cnt++] = 1;
                    numFloats += 1;
                }
            }
        }
    }

    if (f.opt_fog) {
        s += "layout(location=" + std::to_string(cnt) + ") in vec4 aFog;\n";
        s += "layout(location=" + std::to_string(cnt) + ") out vec4 vFog;\n";
        attribSizes[cnt++] = 4;
        numFloats += 4;
    }

    if (f.opt_grayscale) {
        s += "layout(location=" + std::to_string(cnt) + ") in vec4 aGrayscaleColor;\n";
        s += "layout(location=" + std::to_string(cnt) + ") out vec4 vGrayscaleColor;\n";
        attribSizes[cnt++] = 4;
        numFloats += 4;
    }

    for (int i = 0; i < f.numInputs; i++) {
        std::string name = "aInput" + std::to_string(i + 1);
        std::string vname = "vInput" + std::to_string(i + 1);
        if (f.opt_alpha) {
            s += "layout(location=" + std::to_string(cnt) + ") in vec4 " + name + ";\n";
            s += "layout(location=" + std::to_string(cnt) + ") out vec4 " + vname + ";\n";
            attribSizes[cnt++] = 4;
            numFloats += 4;
        } else {
            s += "layout(location=" + std::to_string(cnt) + ") in vec3 " + name + ";\n";
            s += "layout(location=" + std::to_string(cnt) + ") out vec3 " + vname + ";\n";
            attribSizes[cnt++] = 3;
            numFloats += 3;
        }
    }

    s += "void main() {\n";
    for (int i = 0; i < 2; i++) {
        if (f.usedTextures[i]) {
            s += "  vTexCoord" + std::to_string(i) + " = aTexCoord" + std::to_string(i) + ";\n";
            for (int j = 0; j < 2; j++) {
                if (f.clamp[i][j]) {
                    std::string suffix = (j == 0 ? "S" : "T") + std::to_string(i);
                    s += "  vTexClamp" + suffix + " = aTexClamp" + suffix + ";\n";
                }
            }
        }
    }
    if (f.opt_fog) s += "  vFog = aFog;\n";
    if (f.opt_grayscale) s += "  vGrayscaleColor = aGrayscaleColor;\n";
    for (int i = 0; i < f.numInputs; i++) {
        std::string idx = std::to_string(i + 1);
        s += "  vInput" + idx + " = aInput" + idx + ";\n";
    }
    s += "  gl_Position = aVtxPos;\n";
    s += "}\n";

    numFloatsOut = numFloats;
    numAttribsOut = cnt;
    return s;
}

std::string GfxRenderingAPIVulkan::BuildFragmentShader(const CCFeatures& f) {
    std::string s;
    s += "#version 450\n";

    // Push constant block with per-frame/per-draw uniforms
    s += "layout(push_constant) uniform PC {\n";
    s += "  int frameCount;\n";
    s += "  float noiseScale;\n";
    s += "  int textureWidth[2];\n";
    s += "  int textureHeight[2];\n";
    s += "  int textureFiltering[2];\n";
    s += "} pc;\n";

    // Texture samplers with explicit bindings
    if (f.usedTextures[0]) s += "layout(binding=0) uniform sampler2D uTex0;\n";
    if (f.usedTextures[1]) s += "layout(binding=1) uniform sampler2D uTex1;\n";
    if (f.used_masks[0])   s += "layout(binding=2) uniform sampler2D uTexMask0;\n";
    if (f.used_masks[1])   s += "layout(binding=3) uniform sampler2D uTexMask1;\n";
    if (f.used_blend[0])   s += "layout(binding=4) uniform sampler2D uTexBlend0;\n";
    if (f.used_blend[1])   s += "layout(binding=5) uniform sampler2D uTexBlend1;\n";

    // Varyings from vertex shader (locations must match vertex outputs)
    uint8_t loc = 1;
    for (int i = 0; i < 2; i++) {
        if (f.usedTextures[i]) {
            s += "layout(location=" + std::to_string(loc++) + ") in vec2 vTexCoord" + std::to_string(i) + ";\n";
            for (int j = 0; j < 2; j++) {
                if (f.clamp[i][j]) {
                    s += "layout(location=" + std::to_string(loc++) + ") in float vTexClamp" +
                         (j == 0 ? "S" : "T") + std::to_string(i) + ";\n";
                }
            }
        }
    }
    if (f.opt_fog)       s += "layout(location=" + std::to_string(loc++) + ") in vec4 vFog;\n";
    if (f.opt_grayscale) s += "layout(location=" + std::to_string(loc++) + ") in vec4 vGrayscaleColor;\n";
    for (int i = 0; i < f.numInputs; i++) {
        if (f.opt_alpha)
            s += "layout(location=" + std::to_string(loc++) + ") in vec4 vInput" + std::to_string(i + 1) + ";\n";
        else
            s += "layout(location=" + std::to_string(loc++) + ") in vec3 vInput" + std::to_string(i + 1) + ";\n";
    }

    s += "layout(location=0) out vec4 vOutColor;\n";

    s += "#define WRAP(x,low,high) mod((x)-(low),(high)-(low))+(low)\n";
    s += "float random(in vec3 value) {\n";
    s += "  float r=dot(sin(value),vec3(12.9898,78.233,37.719));\n";
    s += "  return fract(sin(r)*143758.5453);\n";
    s += "}\n";
    if (mSrgbMode) {
        s += "vec4 fromLinear(vec4 c){\n";
        s += "  bvec3 cut=lessThan(c.rgb,vec3(0.0031308));\n";
        s += "  return vec4(mix(vec3(1.055)*pow(c.rgb,vec3(1.0/2.4))-vec3(0.055),c.rgb*vec3(12.92),cut),c.a);\n";
        s += "}\n";
    }

    // 3-point filtering helper
    if (mFilterMode == FILTER_THREE_POINT) {
        s += "vec4 filter3pt(sampler2D tex,vec2 uv,vec2 sz){\n";
        s += "  vec2 off=fract(uv*sz-vec2(0.5));\n";
        s += "  off-=step(1.0,off.x+off.y);\n";
        s += "  vec4 c0=texture(tex,uv-off/sz);\n";
        s += "  vec4 c1=texture(tex,uv-vec2(sign(off.x),0)/sz+vec2(-off.x,off.y)/sz);\n";
        s += "  vec4 c2=texture(tex,uv-vec2(0,sign(off.y))/sz+vec2(off.x,-off.y)/sz);\n";
        s += "  return c0+abs(off.x)*(c1-c0)+abs(off.y)*(c2-c0);\n";
        s += "}\n";
    }
    s += "vec4 hookTex(int id,sampler2D tex,vec2 uv,vec2 sz){\n";
    if (mFilterMode == FILTER_THREE_POINT) {
        s += "  if(pc.textureFiltering[id]==" + std::to_string(FILTER_THREE_POINT) + ") return filter3pt(tex,uv,sz);\n";
    }
    s += "  return texture(tex,uv);\n";
    s += "}\n";

    s += "void main() {\n";

    // Texture coordinate clamping + sampling
    for (int i = 0; i < 2; i++) {
        if (f.usedTextures[i]) {
            std::string ti = std::to_string(i);
            bool cs = f.clamp[i][0], ct = f.clamp[i][1];
            s += "  vec2 texSize" + ti + "=vec2(pc.textureWidth[" + ti + "],pc.textureHeight[" + ti + "]);\n";

            if (!cs && !ct) {
                s += "  vec2 vTexCoordAdj" + ti + "=vTexCoord" + ti + ";\n";
            } else if (cs && ct) {
                s += "  vec2 vTexCoordAdj" + ti + "=clamp(vTexCoord" + ti + ",0.5/texSize" + ti +
                     ",vec2(vTexClampS" + ti + ",vTexClampT" + ti + "));\n";
            } else if (cs) {
                s += "  vec2 vTexCoordAdj" + ti + "=vec2(clamp(vTexCoord" + ti + ".s,0.5/texSize" + ti +
                     ".s,vTexClampS" + ti + "),vTexCoord" + ti + ".t);\n";
            } else {
                s += "  vec2 vTexCoordAdj" + ti + "=vec2(vTexCoord" + ti + ".s,clamp(vTexCoord" + ti +
                     ".t,0.5/texSize" + ti + ".t,vTexClampT" + ti + "));\n";
            }

            s += "  vec4 texVal" + ti + "=hookTex(" + ti + ",uTex" + ti + ",vTexCoordAdj" + ti + ",texSize" + ti + ");\n";

            if (f.used_masks[i]) {
                s += "  vec2 maskSize" + ti + "=vec2(textureSize(uTexMask" + ti + ",0));\n";
                s += "  vec4 maskVal" + ti + "=hookTex(" + ti + ",uTexMask" + ti + ",vTexCoordAdj" + ti + ",maskSize" + ti + ");\n";
                if (f.used_blend[i]) {
                    s += "  vec4 blendVal" + ti + "=hookTex(" + ti + ",uTexBlend" + ti + ",vTexCoordAdj" + ti + ",texSize" + ti + ");\n";
                } else {
                    s += "  vec4 blendVal" + ti + "=vec4(0,0,0,0);\n";
                }
                s += "  texVal" + ti + "=mix(texVal" + ti + ",blendVal" + ti + ",maskVal" + ti + ".a);\n";
            }
        }
    }

    // Color combiner
    if (f.opt_alpha) s += "  vec4 texel;\n";
    else             s += "  vec3 texel;\n";

    int fRange = f.opt_2cyc ? 2 : 1;
    for (int cy = 0; cy < fRange; cy++) {
        if (cy == 1) {
            if (f.opt_alpha) {
                if (f.c[cy][1][2] == SHADER_COMBINED)
                    s += "  texel.a=WRAP(texel.a,-1.01,1.01);\n";
                else
                    s += "  texel.a=WRAP(texel.a,-0.51,1.51);\n";
            }
            if (f.c[cy][0][2] == SHADER_COMBINED)
                s += "  texel.rgb=WRAP(texel.rgb,-1.01,1.01);\n";
            else
                s += "  texel.rgb=WRAP(texel.rgb,-0.51,1.51);\n";
        }

        if (!f.color_alpha_same[cy] && f.opt_alpha) {
            s += "  texel=vec4(";
            vk_append_formula(s, f.c[cy], f.do_single[cy][0], f.do_multiply[cy][0], f.do_mix[cy][0],
                              false, false, true, cy == 0);
            s += ",";
            vk_append_formula(s, f.c[cy], f.do_single[cy][1], f.do_multiply[cy][1], f.do_mix[cy][1],
                              true, true, true, cy == 0);
            s += ");\n";
        } else {
            s += "  texel=";
            vk_append_formula(s, f.c[cy], f.do_single[cy][0], f.do_multiply[cy][0], f.do_mix[cy][0],
                              f.opt_alpha, false, f.opt_alpha, cy == 0);
            s += ";\n";
        }
    }

    s += "  texel=WRAP(texel,-0.51,1.51);\n";
    s += "  texel=clamp(texel,0.0,1.0);\n";

    if (f.opt_fog) {
        if (f.opt_alpha)
            s += "  texel=vec4(mix(texel.rgb,vFog.rgb,vFog.a),texel.a);\n";
        else
            s += "  texel=mix(texel,vFog.rgb,vFog.a);\n";
    }

    if (f.opt_texture_edge && f.opt_alpha)
        s += "  if(texel.a>0.19) texel.a=1.0; else discard;\n";

    if (f.opt_alpha && f.opt_noise)
        s += "  texel.a*=floor(clamp(random(vec3(floor(gl_FragCoord.xy*pc.noiseScale),float(pc.frameCount)))+texel.a,0.0,1.0));\n";

    if (f.opt_grayscale) {
        s += "  float intensity=(texel.r+texel.g+texel.b)/3.0;\n";
        s += "  vec3 nt=vGrayscaleColor.rgb*intensity;\n";
        s += "  texel.rgb=mix(texel.rgb,nt,vGrayscaleColor.a);\n";
    }

    if (f.opt_alpha) {
        if (f.opt_alpha_threshold)
            s += "  if(texel.a<8.0/256.0) discard;\n";
        if (f.opt_invisible)
            s += "  texel.a=0.0;\n";
        s += "  vOutColor=texel;\n";
    } else {
        s += "  vOutColor=vec4(texel,1.0);\n";
    }

    if (mSrgbMode)
        s += "  vOutColor=fromLinear(vOutColor);\n";

    s += "}\n";
    return s;
}

// ---- Constructor / Destructor -----------------------------------------------

GfxRenderingAPIVulkan::GfxRenderingAPIVulkan(GfxWindowBackendSDL2* sdlBackend)
    : mSdlBackend(sdlBackend) {
    mShadercOptions.SetOptimizationLevel(shaderc_optimization_level_performance);
    mShadercOptions.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
}

GfxRenderingAPIVulkan::~GfxRenderingAPIVulkan() {
    if (mDevice == VK_NULL_HANDLE) return;

    vkDeviceWaitIdle(mDevice);

    // Destroy framebuffers
    for (auto& fb : mFrameBuffers) DestroyFramebufferImages(fb);

    // Destroy textures
    for (auto& tex : mTextures) {
        if (tex.sampler) vkDestroySampler(mDevice, tex.sampler, nullptr);
        if (tex.view)    vkDestroyImageView(mDevice, tex.view, nullptr);
        if (tex.image)   vmaDestroyImage(mAllocator, tex.image, tex.allocation);
    }

    // Destroy shader programs / pipelines
    for (auto& [key, prg] : mShaderPool) {
        for (auto& p : prg.pipelines)
            if (p) vkDestroyPipeline(mDevice, p, nullptr);
        vkDestroyPipelineLayout(mDevice, prg.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(mDevice, prg.descriptorSetLayout, nullptr);
    }

    // Destroy vertex buffer
    if (mVertexBuffer)
        vmaDestroyBuffer(mAllocator, mVertexBuffer, mVertexBufferAlloc);

    // Destroy frames
    for (auto& fr : mFrames) {
        vkDestroyCommandPool(mDevice, fr.cmdPool, nullptr);
        vkDestroySemaphore(mDevice, fr.imageAvailableSem, nullptr);
        vkDestroySemaphore(mDevice, fr.renderFinishedSem, nullptr);
        vkDestroyFence(mDevice, fr.inFlightFence, nullptr);
    }

    for (int i = 0; i < VULKAN_FRAMES_IN_FLIGHT; i++) {
        vkDestroyDescriptorPool(mDevice, mFrameDescPool[i], nullptr);
    }

    vkDestroyCommandPool(mDevice, mTransferPool, nullptr);
    vkDestroyPipelineCache(mDevice, mPipelineCache, nullptr);
    vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);

    DestroySwapchain();

    vmaDestroyAllocator(mAllocator);
    vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
    vkDestroyDevice(mDevice, nullptr);
    vkDestroyInstance(mInstance, nullptr);
}

// ---- Init / Resize ----------------------------------------------------------

void GfxRenderingAPIVulkan::Init() {
    CreateInstance();
    CreateSurface();
    SelectPhysicalDevice();
    CreateDevice();
    CreateAllocator();
    CreateCommandPools();
    CreateSyncObjects();
    CreateVertexBuffer();
    CreatePipelineCache();
    CreateDescriptorPool();
    CreateSwapchain();

    mFrameBuffers.resize(1); // slot 0 = screen (swapchain)

    // Reserve texture slot 0 so IDs start at 1 (mirrors OpenGL glGenTextures behaviour)
    mTextures.resize(1);

    // ImGui Vulkan backend init must happen after the instance and device are created.
    // Gui::ImGuiBackendInit() is called earlier (before mRapi->Init()), so we do it here.
    VulkanGuiInit();
}

void GfxRenderingAPIVulkan::CreateInstance() {
    unsigned extCount = 0;
    SDL_Vulkan_GetInstanceExtensions(mSdlBackend->GetSDLWindow(), &extCount, nullptr);
    std::vector<const char*> extensions(extCount);
    SDL_Vulkan_GetInstanceExtensions(mSdlBackend->GetSDLWindow(), &extCount, extensions.data());
    // VK_KHR_get_physical_device_properties2 is core since Vulkan 1.1; no need to request it.

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "libultraship";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Fast3D";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();

#ifdef _DEBUG
    // Only request validation if the layer is actually installed
    static const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availLayers.data());
    bool validationAvailable = false;
    for (auto& l : availLayers)
        if (strcmp(l.layerName, kValidationLayer) == 0) { validationAvailable = true; break; }

    if (validationAvailable) {
        ci.enabledLayerCount = 1;
        ci.ppEnabledLayerNames = &kValidationLayer;
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        ci.enabledExtensionCount = (uint32_t)extensions.size();
        ci.ppEnabledExtensionNames = extensions.data();
        SPDLOG_INFO("Vulkan validation layer enabled");
    } else {
        SPDLOG_WARN("VK_LAYER_KHRONOS_validation not found, validation disabled");
    }
#endif

    VK_CHECK(vkCreateInstance(&ci, nullptr, &mInstance));
}

void GfxRenderingAPIVulkan::CreateSurface() {
    if (!SDL_Vulkan_CreateSurface(mSdlBackend->GetSDLWindow(), mInstance, &mSurface)) {
        SPDLOG_ERROR("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        abort();
    }
}

void GfxRenderingAPIVulkan::SelectPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(mInstance, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(mInstance, &count, devs.data());

    // Prefer discrete GPU
    mPhysDevice = devs[0];
    for (auto d : devs) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            mPhysDevice = d;
            break;
        }
    }
}

void GfxRenderingAPIVulkan::CreateDevice() {
    // Find queue families
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysDevice, &qfCount, qfProps.data());

    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            mGraphicsQueueFamily = i;

        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(mPhysDevice, i, mSurface, &present);
        if (present) mPresentQueueFamily = i;

        if (mGraphicsQueueFamily != UINT32_MAX && mPresentQueueFamily != UINT32_MAX) break;
    }

    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    auto addQ = [&](uint32_t fam) {
        for (auto& q : qcis) if (q.queueFamilyIndex == fam) return;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = fam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        qcis.push_back(qci);
    };
    addQ(mGraphicsQueueFamily);
    addQ(mPresentQueueFamily);

    // Swapchain is still an extension in Vulkan 1.3; dynamic rendering and
    // extended dynamic state are now core, so no KHR/EXT extensions needed.
    const char* devExts[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    // Vulkan 1.3 unified feature struct covers dynamic rendering, sync2, etc.
    VkPhysicalDeviceVulkan13Features vk13{};
    vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vk13.dynamicRendering = VK_TRUE;
    vk13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceFeatures2 feat2{};
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.features.depthClamp = VK_TRUE;
    feat2.pNext = &vk13;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &feat2;
    dci.queueCreateInfoCount = (uint32_t)qcis.size();
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = (uint32_t)std::size(devExts);
    dci.ppEnabledExtensionNames = devExts;

    VK_CHECK(vkCreateDevice(mPhysDevice, &dci, nullptr, &mDevice));
    vkGetDeviceQueue(mDevice, mGraphicsQueueFamily, 0, &mGraphicsQueue);
    vkGetDeviceQueue(mDevice, mPresentQueueFamily, 0, &mPresentQueue);
}

void GfxRenderingAPIVulkan::CreateAllocator() {
    VmaAllocatorCreateInfo ai{};
    ai.physicalDevice = mPhysDevice;
    ai.device = mDevice;
    ai.instance = mInstance;
    ai.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&ai, &mAllocator));
}

void GfxRenderingAPIVulkan::CreateCommandPools() {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = mGraphicsQueueFamily;

    for (auto& fr : mFrames) {
        VK_CHECK(vkCreateCommandPool(mDevice, &ci, nullptr, &fr.cmdPool));
        VkCommandBufferAllocateInfo ai2{};
        ai2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai2.commandPool = fr.cmdPool;
        ai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai2.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(mDevice, &ai2, &fr.cmdBuf));
    }

    VkCommandPoolCreateInfo tci{};
    tci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    tci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    tci.queueFamilyIndex = mGraphicsQueueFamily;
    VK_CHECK(vkCreateCommandPool(mDevice, &tci, nullptr, &mTransferPool));
}

void GfxRenderingAPIVulkan::CreateSyncObjects() {
    VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (auto& fr : mFrames) {
        VK_CHECK(vkCreateSemaphore(mDevice, &si, nullptr, &fr.imageAvailableSem));
        VK_CHECK(vkCreateSemaphore(mDevice, &si, nullptr, &fr.renderFinishedSem));
        VK_CHECK(vkCreateFence(mDevice, &fi, nullptr, &fr.inFlightFence));
    }
}

void GfxRenderingAPIVulkan::CreateVertexBuffer() {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = VERTEX_BUFFER_SIZE;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc{};
    alloc.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(mAllocator, &bci, &alloc, &mVertexBuffer, &mVertexBufferAlloc, &info));
    mVertexBufferMapped = info.pMappedData;
}

void GfxRenderingAPIVulkan::CreatePipelineCache() {
    VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    VK_CHECK(vkCreatePipelineCache(mDevice, &ci, nullptr, &mPipelineCache));
}

void GfxRenderingAPIVulkan::CreateDescriptorPool() {
    // Large pool for ImGui + per-shader descriptor sets
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096 },
    };
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 1024;
    ci.poolSizeCount = (uint32_t)std::size(poolSizes);
    ci.pPoolSizes = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(mDevice, &ci, nullptr, &mDescriptorPool));

    // Per-frame descriptor pools (reset each frame)
    VkDescriptorPoolSize fpSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 },
    };
    for (int i = 0; i < VULKAN_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorPoolCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        fci.maxSets = 512;
        fci.poolSizeCount = (uint32_t)std::size(fpSizes);
        fci.pPoolSizes = fpSizes;
        VK_CHECK(vkCreateDescriptorPool(mDevice, &fci, nullptr, &mFrameDescPool[i]));
    }
}

void GfxRenderingAPIVulkan::CreateSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysDevice, mSurface, &caps);

    // Format selection
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysDevice, mSurface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysDevice, mSurface, &fmtCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& fmt : formats) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = fmt;
            break;
        }
        if (fmt.format == VK_FORMAT_B8G8R8A8_UNORM) chosen = fmt;
    }
    mSwapchainFormat = chosen.format;

    // Present mode
    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysDevice, mSurface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysDevice, mSurface, &pmCount, presentModes.data());

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    bool vsync = Ship::Context::GetInstance()->GetConsoleVariables()->GetInteger(CVAR_VSYNC_ENABLED, 1);
    if (!vsync) {
        for (auto pm : presentModes) {
            if (pm == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = pm; break; }
            if (pm == VK_PRESENT_MODE_IMMEDIATE_KHR) presentMode = pm;
        }
    }

    // Extent
    if (caps.currentExtent.width != UINT32_MAX) {
        mSwapchainExtent = caps.currentExtent;
    } else {
        int w, h;
        SDL_Vulkan_GetDrawableSize(mSdlBackend->GetSDLWindow(), &w, &h);
        mSwapchainExtent.width = std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width);
        mSwapchainExtent.height = std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = mSurface;
    sci.minImageCount = imageCount;
    sci.imageFormat = mSwapchainFormat;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = mSwapchainExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = presentMode;
    sci.clipped = VK_TRUE;

    uint32_t qfIndices[] = { mGraphicsQueueFamily, mPresentQueueFamily };
    if (mGraphicsQueueFamily != mPresentQueueFamily) {
        sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2;
        sci.pQueueFamilyIndices = qfIndices;
    } else {
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VK_CHECK(vkCreateSwapchainKHR(mDevice, &sci, nullptr, &mSwapchain));

    uint32_t imgCount = 0;
    vkGetSwapchainImagesKHR(mDevice, mSwapchain, &imgCount, nullptr);
    mSwapchainImages.resize(imgCount);
    vkGetSwapchainImagesKHR(mDevice, mSwapchain, &imgCount, mSwapchainImages.data());

    mSwapchainImageViews.resize(imgCount);
    for (uint32_t i = 0; i < imgCount; i++) {
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = mSwapchainImages[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = mSwapchainFormat;
        ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(mDevice, &ivci, nullptr, &mSwapchainImageViews[i]));
    }
}

void GfxRenderingAPIVulkan::DestroySwapchain() {
    for (auto iv : mSwapchainImageViews) vkDestroyImageView(mDevice, iv, nullptr);
    mSwapchainImageViews.clear();
    mSwapchainImages.clear();
    if (mSwapchain) { vkDestroySwapchainKHR(mDevice, mSwapchain, nullptr); mSwapchain = VK_NULL_HANDLE; }
}

void GfxRenderingAPIVulkan::OnResize() {
    if (mDevice == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(mDevice);
    DestroySwapchain();
    CreateSwapchain();
}

// ---- Frame management -------------------------------------------------------

void GfxRenderingAPIVulkan::StartFrame() {
    mFrameCount++;

    auto& fr = mFrames[mCurrentFrame];

    // Wait for this frame's fence
    VK_CHECK(vkWaitForFences(mDevice, 1, &fr.inFlightFence, VK_TRUE, UINT64_MAX));

    // Acquire swapchain image
    VkResult res = vkAcquireNextImageKHR(mDevice, mSwapchain, UINT64_MAX,
                                          fr.imageAvailableSem, VK_NULL_HANDLE, &mSwapchainImageIndex);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        OnResize();
        res = vkAcquireNextImageKHR(mDevice, mSwapchain, UINT64_MAX,
                                     fr.imageAvailableSem, VK_NULL_HANDLE, &mSwapchainImageIndex);
    }
    VK_CHECK(res);

    VK_CHECK(vkResetFences(mDevice, 1, &fr.inFlightFence));

    // Reset per-frame descriptor pool
    vkResetDescriptorPool(mDevice, mFrameDescPool[mCurrentFrame], 0);
    mFrameDescSetIdx[mCurrentFrame] = 0;

    // Begin command buffer
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandBuffer(fr.cmdBuf, 0));
    VK_CHECK(vkBeginCommandBuffer(fr.cmdBuf, &bi));

    // Transition swapchain image to color attachment
    TransitionImageLayout(fr.cmdBuf, mSwapchainImages[mSwapchainImageIndex],
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Enable scissor test by default
    VkRect2D scissor{ {0,0}, mSwapchainExtent };
    vkCmdSetScissor(fr.cmdBuf, 0, 1, &scissor);

    mFrameStarted = true;
}

void GfxRenderingAPIVulkan::EndFrame() {
    if (!mFrameStarted) return;

    auto& fr = mFrames[mCurrentFrame];

    // End any active rendering pass
    // (EndDraw called separately from StartDrawToFramebuffer)

    // Transition swapchain image to present
    TransitionImageLayout(fr.cmdBuf, mSwapchainImages[mSwapchainImageIndex],
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(fr.cmdBuf));

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &fr.imageAvailableSem;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &fr.cmdBuf;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &fr.renderFinishedSem;
    VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &si, fr.inFlightFence));

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &fr.renderFinishedSem;
    pi.swapchainCount = 1;
    pi.pSwapchains = &mSwapchain;
    pi.pImageIndices = &mSwapchainImageIndex;
    VkResult res = vkQueuePresentKHR(mPresentQueue, &pi);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) OnResize();

    mCurrentFrame = (mCurrentFrame + 1) % VULKAN_FRAMES_IN_FLIGHT;
    mFrameStarted = false;
}

void GfxRenderingAPIVulkan::FinishRender() {
    // Nothing needed here; EndFrame handles submission
}

// ---- Framebuffer management -------------------------------------------------

int GfxRenderingAPIVulkan::CreateFramebuffer() {
    size_t i = mFrameBuffers.size();
    mFrameBuffers.emplace_back();
    return (int)i;
}

void GfxRenderingAPIVulkan::RecreateFramebufferImages(VulkanFramebuffer& fb) {
    DestroyFramebufferImages(fb);

    if (fb.width == 0 || fb.height == 0) return;

    VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
    if (fb.msaaLevel > 1) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(mPhysDevice, &props);
        uint32_t maxMsaa = props.limits.framebufferColorSampleCounts;
        if (maxMsaa >= 8 && fb.msaaLevel >= 8) sampleCount = VK_SAMPLE_COUNT_8_BIT;
        else if (maxMsaa >= 4 && fb.msaaLevel >= 4) sampleCount = VK_SAMPLE_COUNT_4_BIT;
        else sampleCount = VK_SAMPLE_COUNT_2_BIT;
    }

    auto createImage = [&](uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                           VkSampleCountFlagBits samples, VkImage& img, VmaAllocation& alloc, VkImageView& view,
                           VkImageAspectFlags aspect) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = { w, h, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = samples;
        ici.usage = usage;

        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(mAllocator, &ici, &vaci, &img, &alloc, nullptr));

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = img;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = fmt;
        ivci.subresourceRange = { aspect, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(mDevice, &ivci, nullptr, &view));
    };

    fb.colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    if (fb.msaaLevel > 1) {
        createImage(fb.width, fb.height, fb.colorFormat, colorUsage | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                    sampleCount, fb.colorImage, fb.colorAlloc, fb.colorView, VK_IMAGE_ASPECT_COLOR_BIT);

        // resolve target
        createImage(fb.width, fb.height, fb.colorFormat, colorUsage,
                    VK_SAMPLE_COUNT_1_BIT, fb.colorResolvImage, fb.colorResolvAlloc, fb.colorResolvView,
                    VK_IMAGE_ASPECT_COLOR_BIT);
    } else {
        createImage(fb.width, fb.height, fb.colorFormat, colorUsage,
                    VK_SAMPLE_COUNT_1_BIT, fb.colorImage, fb.colorAlloc, fb.colorView, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    if (fb.hasDepth) {
        createImage(fb.width, fb.height, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    sampleCount, fb.depthImage, fb.depthAlloc, fb.depthView, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Transition new images to expected layouts
    VkCommandBuffer cmd = BeginOneTimeSubmit();
    if (fb.msaaLevel > 1) {
        TransitionImageLayout(cmd, fb.colorResolvImage, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    TransitionImageLayout(cmd, fb.colorImage, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (fb.hasDepth)
        TransitionImageLayout(cmd, fb.depthImage, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_ASPECT_DEPTH_BIT);
    EndOneTimeSubmit(cmd);
}

void GfxRenderingAPIVulkan::DestroyFramebufferImages(VulkanFramebuffer& fb) {
    if (fb.colorView)       { vkDestroyImageView(mDevice, fb.colorView, nullptr);       fb.colorView = VK_NULL_HANDLE; }
    if (fb.colorImage)      { vmaDestroyImage(mAllocator, fb.colorImage, fb.colorAlloc); fb.colorImage = VK_NULL_HANDLE; }
    if (fb.colorResolvView) { vkDestroyImageView(mDevice, fb.colorResolvView, nullptr);  fb.colorResolvView = VK_NULL_HANDLE; }
    if (fb.colorResolvImage){ vmaDestroyImage(mAllocator, fb.colorResolvImage, fb.colorResolvAlloc); fb.colorResolvImage = VK_NULL_HANDLE; }
    if (fb.depthView)       { vkDestroyImageView(mDevice, fb.depthView, nullptr);        fb.depthView = VK_NULL_HANDLE; }
    if (fb.depthImage)      { vmaDestroyImage(mAllocator, fb.depthImage, fb.depthAlloc); fb.depthImage = VK_NULL_HANDLE; }
}

void GfxRenderingAPIVulkan::UpdateFramebufferParameters(int fbId, uint32_t width, uint32_t height,
                                                         uint32_t msaaLevel, bool oglInvertY,
                                                         bool renderTarget, bool hasDepth,
                                                         bool /*canExtractDepth*/) {
    if (fbId == 0) return; // screen framebuffer managed by swapchain

    auto& fb = mFrameBuffers[fbId];
    width = std::max(width, 1U);
    height = std::max(height, 1U);

    if (fb.width == width && fb.height == height && fb.msaaLevel == msaaLevel &&
        fb.hasDepth == hasDepth) return;

    vkDeviceWaitIdle(mDevice);
    fb.width = width;
    fb.height = height;
    fb.msaaLevel = msaaLevel;
    fb.hasDepth = hasDepth;
    fb.invertY = oglInvertY;
    RecreateFramebufferImages(fb);
}

void GfxRenderingAPIVulkan::StartDrawToFramebuffer(int fbId, float noiseScale) {
    mCurrentFbId = fbId;
    if (noiseScale != 0.0f) mCurrentNoiseScale = 1.0f / noiseScale;

    auto& fr = mFrames[mCurrentFrame];
    VkCommandBuffer cmd = fr.cmdBuf;

    VkRenderingAttachmentInfo colorAttach{};
    colorAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAttach{};
    depthAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAttach;

    if (fbId == 0) {
        colorAttach.imageView = mSwapchainImageViews[mSwapchainImageIndex];
        colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ri.renderArea = { {0,0}, mSwapchainExtent };
        ri.layerCount = 1;
        vkCmdBeginRendering(cmd, &ri);
    } else {
        auto& fb = mFrameBuffers[fbId];
        colorAttach.imageView = fb.colorView;
        colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ri.renderArea = { {0,0}, {fb.width, fb.height} };
        ri.layerCount = 1;
        if (fb.hasDepth) {
            depthAttach.imageView = fb.depthView;
            depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            ri.pDepthAttachment = &depthAttach;
        }
        vkCmdBeginRendering(cmd, &ri);
    }
}

void GfxRenderingAPIVulkan::ClearFramebuffer(bool color, bool depth) {
    auto& fr = mFrames[mCurrentFrame];
    VkCommandBuffer cmd = fr.cmdBuf;

    std::vector<VkClearAttachment> clears;
    if (color) {
        VkClearAttachment ca{};
        ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ca.colorAttachment = 0;
        ca.clearValue.color = { {0,0,0,1} };
        clears.push_back(ca);
    }
    if (depth && mCurrentFbId != 0) {
        auto& fb = mFrameBuffers[mCurrentFbId];
        if (fb.hasDepth) {
            VkClearAttachment da{};
            da.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            da.clearValue.depthStencil = { 1.0f, 0 };
            clears.push_back(da);
        }
    }

    if (clears.empty()) return;

    VkClearRect rect{};
    rect.rect = { {0,0}, mCurrentFbId == 0 ? mSwapchainExtent : VkExtent2D{mFrameBuffers[mCurrentFbId].width, mFrameBuffers[mCurrentFbId].height} };
    rect.layerCount = 1;
    vkCmdClearAttachments(cmd, (uint32_t)clears.size(), clears.data(), 1, &rect);
}

void GfxRenderingAPIVulkan::CopyFramebuffer(int fbDstId, int fbSrcId, int sx0, int sy0, int sx1, int sy1,
                                             int dx0, int dy0, int dx1, int dy1) {
    // End any active rendering before blit
    auto& fr = mFrames[mCurrentFrame];
    VkCommandBuffer cmd = fr.cmdBuf;
    vkCmdEndRendering(cmd);

    if (fbSrcId >= (int)mFrameBuffers.size() || fbDstId >= (int)mFrameBuffers.size()) return;

    auto& src = mFrameBuffers[fbSrcId];
    auto& dst = mFrameBuffers[fbDstId];

    VkImage srcImg = fbSrcId == 0 ? mSwapchainImages[mSwapchainImageIndex] : src.colorImage;
    VkImage dstImg = fbDstId == 0 ? mSwapchainImages[mSwapchainImageIndex] : dst.colorImage;

    // Transition for blit
    TransitionImageLayout(cmd, srcImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    TransitionImageLayout(cmd, dstImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageBlit blit{};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.srcOffsets[0] = { sx0, sy0, 0 };
    blit.srcOffsets[1] = { sx1, sy1, 1 };
    blit.dstOffsets[0] = { dx0, dy0, 0 };
    blit.dstOffsets[1] = { dx1, dy1, 1 };
    vkCmdBlitImage(cmd, srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

    // Transition back
    TransitionImageLayout(cmd, srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImageLayout(cmd, dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}

void GfxRenderingAPIVulkan::ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) {
    // Create readback staging buffer
    VkDeviceSize size = (VkDeviceSize)width * height * 4;
    VkBuffer stageBuf;
    VmaAllocation stageAlloc;
    VmaAllocationInfo stageInfo{};
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VK_CHECK(vmaCreateBuffer(mAllocator, &bci, &vaci, &stageBuf, &stageAlloc, &stageInfo));

    VkCommandBuffer cmd = BeginOneTimeSubmit();
    VkImage srcImg = fbId == 0 ? mSwapchainImages[mSwapchainImageIndex] : mFrameBuffers[fbId].colorImage;
    TransitionImageLayout(cmd, srcImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };
    vkCmdCopyImageToBuffer(cmd, srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stageBuf, 1, &region);
    TransitionImageLayout(cmd, srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    EndOneTimeSubmit(cmd);

    // Convert RGBA8 → RGBA16
    const uint8_t* src = (const uint8_t*)stageInfo.pMappedData;
    for (uint32_t i = 0; i < width * height; i++) {
        uint8_t r = src[i * 4 + 0], g = src[i * 4 + 1], b = src[i * 4 + 2], a = src[i * 4 + 3];
        rgba16Buf[i] = ((r >> 3) << 11) | ((g >> 3) << 6) | ((b >> 3) << 1) | (a >> 7);
    }
    vmaDestroyBuffer(mAllocator, stageBuf, stageAlloc);
}

void GfxRenderingAPIVulkan::ResolveMSAAColorBuffer(int fbIdTarget, int fbIdSrc) {
    if (fbIdSrc >= (int)mFrameBuffers.size() || fbIdTarget >= (int)mFrameBuffers.size()) return;

    auto& fr = mFrames[mCurrentFrame];
    VkCommandBuffer cmd = fr.cmdBuf;
    vkCmdEndRendering(cmd);

    auto& src = mFrameBuffers[fbIdSrc];
    auto& dst = mFrameBuffers[fbIdTarget];
    if (src.msaaLevel <= 1 || src.colorResolvImage == VK_NULL_HANDLE) return;

    TransitionImageLayout(cmd, src.colorImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    TransitionImageLayout(cmd, dst.colorImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageResolve resolve{};
    resolve.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    resolve.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    resolve.extent = { src.width, src.height, 1 };
    vkCmdResolveImage(cmd, src.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      dst.colorImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &resolve);

    TransitionImageLayout(cmd, src.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    TransitionImageLayout(cmd, dst.colorImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}

// ---- Texture management -----------------------------------------------------

uint32_t GfxRenderingAPIVulkan::NewTexture() {
    uint32_t id = (uint32_t)mTextures.size();
    mTextures.emplace_back();
    return id;
}

void GfxRenderingAPIVulkan::DeleteTexture(uint32_t texId) {
    if (texId >= mTextures.size()) return;
    auto& tex = mTextures[texId];
    vkDeviceWaitIdle(mDevice);
    if (tex.sampler) vkDestroySampler(mDevice, tex.sampler, nullptr);
    if (tex.view)    vkDestroyImageView(mDevice, tex.view, nullptr);
    if (tex.image)   vmaDestroyImage(mAllocator, tex.image, tex.allocation);
    tex = {};
}

void GfxRenderingAPIVulkan::SelectTexture(int tile, uint32_t textureId) {
    mCurrentTextureIds[tile] = textureId;
    mCurrentTile = tile;
}

void GfxRenderingAPIVulkan::UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;

    uint32_t texId = mCurrentTextureIds[mCurrentTile];
    auto& tex = mTextures[texId];

    // Destroy old image if size changed
    if (tex.image && (tex.width != width || tex.height != height)) {
        vkDeviceWaitIdle(mDevice);
        if (tex.view)  vkDestroyImageView(mDevice, tex.view, nullptr);
        vmaDestroyImage(mAllocator, tex.image, tex.allocation);
        tex.image = VK_NULL_HANDLE;
        tex.view = VK_NULL_HANDLE;
    }

    tex.width = width;
    tex.height = height;

    // Create image if needed
    if (!tex.image) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = { width, height, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo vaci{};
        vaci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(mAllocator, &ici, &vaci, &tex.image, &tex.allocation, nullptr));

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = tex.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(mDevice, &ivci, nullptr, &tex.view));
    }

    // Upload via staging buffer
    VkDeviceSize dataSize = (VkDeviceSize)width * height * 4;
    VkBuffer stageBuf;
    VmaAllocation stageAlloc;
    VmaAllocationInfo stageInfo{};
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = dataSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VK_CHECK(vmaCreateBuffer(mAllocator, &bci, &vaci, &stageBuf, &stageAlloc, &stageInfo));
    memcpy(stageInfo.pMappedData, rgba32Buf, dataSize);

    VkCommandBuffer cmd = BeginOneTimeSubmit();
    TransitionImageLayout(cmd, tex.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { width, height, 1 };
    vkCmdCopyBufferToImage(cmd, stageBuf, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    TransitionImageLayout(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    EndOneTimeSubmit(cmd);
    vmaDestroyBuffer(mAllocator, stageBuf, stageAlloc);

    // (Re)create sampler
    if (tex.sampler) vkDestroySampler(mDevice, tex.sampler, nullptr);
    tex.sampler = GetOrCreateSampler(tex);
}

static VkSamplerAddressMode cm_to_vk(uint32_t val) {
    switch (val) {
        case G_TX_NOMIRROR | G_TX_CLAMP:  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case G_TX_MIRROR | G_TX_WRAP:     return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case G_TX_MIRROR | G_TX_CLAMP:    return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        case G_TX_NOMIRROR | G_TX_WRAP:   return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        default:                          return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

void GfxRenderingAPIVulkan::SetSamplerParameters(int tile, bool linearFilter, uint32_t cms, uint32_t cmt) {
    uint32_t texId = mCurrentTextureIds[tile];
    if (texId >= mTextures.size()) return;
    auto& tex = mTextures[texId];

    VkFilter filter = (linearFilter && mFilterMode == FILTER_LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    tex.linearFiltering = !linearFilter; // mirrors OGL backend's logic
    tex.minFilter = filter;
    tex.magFilter = filter;
    tex.wrapS = cm_to_vk(cms);
    tex.wrapT = cm_to_vk(cmt);

    if (tex.sampler) {
        vkDestroySampler(mDevice, tex.sampler, nullptr);
        tex.sampler = VK_NULL_HANDLE;
    }
    if (tex.image) tex.sampler = GetOrCreateSampler(tex);
}

VkSampler GfxRenderingAPIVulkan::GetOrCreateSampler(const VulkanTexture& tex) {
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = tex.magFilter;
    sci.minFilter = tex.minFilter;
    sci.addressModeU = tex.wrapS;
    sci.addressModeV = tex.wrapT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    VkSampler s;
    VK_CHECK(vkCreateSampler(mDevice, &sci, nullptr, &s));
    return s;
}

void GfxRenderingAPIVulkan::SelectTextureFb(int fbId) {
    // Use the off-screen FB's color image as texture slot 0
    // We keep a "virtual" texture entry backed by the FB image
    uint32_t texId;
    if (fbId < (int)mFrameBuffers.size()) {
        // Allocate a pseudo-texture slot for the FB if needed
        // FB textures share slots starting past the normal texture range
        // Simple approach: use fbId offset by a large number
        texId = (uint32_t)(fbId + 0x4000);
        if (texId >= mTextures.size()) mTextures.resize(texId + 1);
        auto& tex = mTextures[texId];
        auto& fb = mFrameBuffers[fbId];
        tex.view = fbId == 0 ? mSwapchainImageViews[mSwapchainImageIndex]
                             : (fb.msaaLevel > 1 && fb.colorResolvView ? fb.colorResolvView : fb.colorView);
        tex.width = fbId == 0 ? mSwapchainExtent.width : fb.width;
        tex.height = fbId == 0 ? mSwapchainExtent.height : fb.height;
        if (!tex.sampler) tex.sampler = GetOrCreateSampler(tex);
    } else {
        return;
    }
    SelectTexture(0, texId);
}

// ---- Shader / Pipeline management -------------------------------------------

VkShaderModule GfxRenderingAPIVulkan::CompileShader(const std::string& source, shaderc_shader_kind kind,
                                                     const char* name) {
    auto result = mShadercCompiler.CompileGlslToSpv(source, kind, name, mShadercOptions);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        SPDLOG_ERROR("Shader compile error in {}: {}", name, result.GetErrorMessage());
        SPDLOG_ERROR("Source:\n{}", source);
        abort();
    }
    std::vector<uint32_t> spv(result.begin(), result.end());

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size() * sizeof(uint32_t);
    smci.pCode = spv.data();
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(mDevice, &smci, nullptr, &mod));
    return mod;
}


ShaderProgramVulkan* GfxRenderingAPIVulkan::BuildShaderProgram(uint64_t shaderId0, uint64_t shaderId1) {
    CCFeatures f;
    gfx_cc_get_features(shaderId0, shaderId1, &f);

    uint8_t numFloats = 0, numAttribs = 0;
    uint8_t attribSizes[16] = {};
    const std::string vsSource = BuildVertexShader(f, numFloats, numAttribs, attribSizes);
    const std::string fsSource = BuildFragmentShader(f);

    VkShaderModule vsMod = CompileShader(vsSource, shaderc_vertex_shader, "vertex");
    VkShaderModule fsMod = CompileShader(fsSource, shaderc_fragment_shader, "fragment");

    // Descriptor set layout (texture samplers)
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    auto addBinding = [&](uint32_t binding) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = binding;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(b);
    };
    if (f.usedTextures[0]) addBinding(0);
    if (f.usedTextures[1]) addBinding(1);
    if (f.used_masks[0])   addBinding(2);
    if (f.used_masks[1])   addBinding(3);
    if (f.used_blend[0])   addBinding(4);
    if (f.used_blend[1])   addBinding(5);

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = (uint32_t)bindings.size();
    dslci.pBindings = bindings.data();

    auto& prg = mShaderPool[{shaderId0, shaderId1}];
    VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &dslci, nullptr, &prg.descriptorSetLayout));

    // Pipeline layout
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = bindings.empty() ? 0 : 1;
    plci.pSetLayouts = &prg.descriptorSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(mDevice, &plci, nullptr, &prg.pipelineLayout));

    // Vertex input descriptions
    std::vector<VkVertexInputAttributeDescription> attrs;
    uint32_t offset = 0;
    for (uint8_t i = 0; i < numAttribs; i++) {
        VkVertexInputAttributeDescription ad{};
        ad.location = i;
        ad.binding = 0;
        ad.offset = offset * sizeof(float);
        switch (attribSizes[i]) {
            case 1: ad.format = VK_FORMAT_R32_SFLOAT; break;
            case 2: ad.format = VK_FORMAT_R32G32_SFLOAT; break;
            case 3: ad.format = VK_FORMAT_R32G32B32_SFLOAT; break;
            case 4: ad.format = VK_FORMAT_R32G32B32A32_SFLOAT; break;
        }
        attrs.push_back(ad);
        offset += attribSizes[i];
    }

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = numFloats * sizeof(float);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vsMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fsMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo visci{};
    visci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (!attrs.empty()) {
        visci.vertexBindingDescriptionCount = 1;
        visci.pVertexBindingDescriptions = &binding;
        visci.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
        visci.pVertexAttributeDescriptions = attrs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo iasci{};
    iasci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vsci{};
    vsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vsci.viewportCount = 1;
    vsci.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rsci{};
    rsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsci.polygonMode = VK_POLYGON_MODE_FILL;
    rsci.cullMode = VK_CULL_MODE_NONE;
    rsci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsci.lineWidth = 1.0f;
    rsci.depthBiasEnable = VK_TRUE; // always enabled; use vkCmdSetDepthBias to zero when not decal

    VkPipelineMultisampleStateCreateInfo msci{};
    msci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dssci{};
    dssci.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    // depth test/write/compareOp are dynamic

    // Color blend (one per pipeline variant: blend off [0] and blend on [1])
    VkFormat colorFmt = mCurrentFbId == 0 ? mSwapchainFormat : (mCurrentFbId < (int)mFrameBuffers.size() ? mFrameBuffers[mCurrentFbId].colorFormat : mSwapchainFormat);
    VkFormat depthFmt = VK_FORMAT_UNDEFINED;
    if (mCurrentFbId != 0 && mCurrentFbId < (int)mFrameBuffers.size() && mFrameBuffers[mCurrentFbId].hasDepth)
        depthFmt = VK_FORMAT_D32_SFLOAT;

    std::vector<VkDynamicState> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    VkPipelineDynamicStateCreateInfo dynci{};
    dynci.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynci.dynamicStateCount = (uint32_t)dynStates.size();
    dynci.pDynamicStates = dynStates.data();

    // Dynamic rendering info
    VkPipelineRenderingCreateInfo prc{};
    prc.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prc.colorAttachmentCount = 1;
    prc.pColorAttachmentFormats = &colorFmt;
    prc.depthAttachmentFormat = depthFmt;

    for (int blend = 0; blend < 2; blend++) {
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (blend == 1) {
            cba.blendEnable = VK_TRUE;
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cba.colorBlendOp = VK_BLEND_OP_ADD;
            cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cba.alphaBlendOp = VK_BLEND_OP_ADD;
        }

        VkPipelineColorBlendStateCreateInfo cbsci{};
        cbsci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cbsci.attachmentCount = 1;
        cbsci.pAttachments = &cba;

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext = &prc;
        gpci.stageCount = 2;
        gpci.pStages = stages;
        gpci.pVertexInputState = &visci;
        gpci.pInputAssemblyState = &iasci;
        gpci.pViewportState = &vsci;
        gpci.pRasterizationState = &rsci;
        gpci.pMultisampleState = &msci;
        gpci.pDepthStencilState = &dssci;
        gpci.pColorBlendState = &cbsci;
        gpci.pDynamicState = &dynci;
        gpci.layout = prg.pipelineLayout;

        VK_CHECK(vkCreateGraphicsPipelines(mDevice, mPipelineCache, 1, &gpci, nullptr, &prg.pipelines[blend]));
    }

    vkDestroyShaderModule(mDevice, vsMod, nullptr);
    vkDestroyShaderModule(mDevice, fsMod, nullptr);

    prg.numInputs = f.numInputs;
    prg.usedTextures[0] = f.usedTextures[0];
    prg.usedTextures[1] = f.usedTextures[1];
    prg.usedTextures[2] = f.used_masks[0];
    prg.usedTextures[3] = f.used_masks[1];
    prg.usedTextures[4] = f.used_blend[0];
    prg.usedTextures[5] = f.used_blend[1];
    prg.numFloats = numFloats;
    prg.numAttribs = numAttribs;
    memcpy(prg.attribSizes, attribSizes, sizeof(attribSizes));

    return &prg;
}

ShaderProgram* GfxRenderingAPIVulkan::CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto* prg = BuildShaderProgram(shaderId0, shaderId1);
    LoadShader((ShaderProgram*)prg);
    return (ShaderProgram*)prg;
}

ShaderProgram* GfxRenderingAPIVulkan::LookupShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto it = mShaderPool.find({shaderId0, shaderId1});
    return it == mShaderPool.end() ? nullptr : (ShaderProgram*)&it->second;
}

void GfxRenderingAPIVulkan::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    auto* p = (ShaderProgramVulkan*)prg;
    *numInputs = p->numInputs;
    usedTextures[0] = p->usedTextures[0];
    usedTextures[1] = p->usedTextures[1];
}

void GfxRenderingAPIVulkan::LoadShader(ShaderProgram* prg) {
    mCurrentShader = (ShaderProgramVulkan*)prg;
}

void GfxRenderingAPIVulkan::UnloadShader(ShaderProgram* /*oldPrg*/) {
    mCurrentShader = nullptr;
}

// ---- Draw state + triangle drawing ------------------------------------------

void GfxRenderingAPIVulkan::SetDepthTestAndMask(bool depthTest, bool zUpd) {
    mDepthTestEnabled = depthTest;
    mDepthWriteEnabled = zUpd;
}

void GfxRenderingAPIVulkan::SetZmodeDecal(bool decal) {
    mZmodeDecal = decal;
}

void GfxRenderingAPIVulkan::SetViewport(int x, int y, int width, int height) {
    auto& fr = mFrames[mCurrentFrame];
    VkViewport vp{};
    vp.x = (float)x;
    // Vulkan Y is top-down but framebuffer 0 may need flip.
    // For simplicity: use standard top-down coordinates.
    vp.y = (float)y;
    vp.width = (float)width;
    vp.height = (float)height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(fr.cmdBuf, 0, 1, &vp);
}

void GfxRenderingAPIVulkan::SetScissor(int x, int y, int width, int height) {
    auto& fr = mFrames[mCurrentFrame];
    VkRect2D scissor{ {x, y}, {(uint32_t)std::max(0, width), (uint32_t)std::max(0, height)} };
    vkCmdSetScissor(fr.cmdBuf, 0, 1, &scissor);
}

void GfxRenderingAPIVulkan::SetUseAlpha(bool useAlpha) {
    mBlendEnabled = useAlpha;
}

void GfxRenderingAPIVulkan::DrawTriangles(float bufVbo[], size_t bufVboLen, size_t bufVboNumTris) {
    if (!mCurrentShader) return;

    auto& fr = mFrames[mCurrentFrame];
    VkCommandBuffer cmd = fr.cmdBuf;

    // Bind pipeline
    VkPipeline pipe = mCurrentShader->pipelines[mBlendEnabled ? 1 : 0];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

    // Set dynamic depth state
    vkCmdSetDepthTestEnable(cmd, mDepthTestEnabled || mDepthWriteEnabled ? VK_TRUE : VK_FALSE);
    vkCmdSetDepthWriteEnable(cmd, mDepthWriteEnabled ? VK_TRUE : VK_FALSE);
    vkCmdSetDepthCompareOp(cmd, mDepthTestEnabled
        ? (mZmodeDecal ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_LESS)
        : VK_COMPARE_OP_ALWAYS);

    // Depth bias for decal mode
    if (mZmodeDecal) {
        float height = (mCurrentFbId < (int)mFrameBuffers.size() && mCurrentFbId != 0)
            ? (float)mFrameBuffers[mCurrentFbId].height : (float)mSwapchainExtent.height;
        float ssdb = -1.0f * height / 120.0f;
        vkCmdSetDepthBias(cmd, ssdb, 0.0f, ssdb);
    } else {
        vkCmdSetDepthBias(cmd, 0.0f, 0.0f, 0.0f);
    }

    // Upload push constants
    PushConstants pc{};
    pc.frameCount = (int32_t)mFrameCount;
    pc.noiseScale = mCurrentNoiseScale;
    for (int i = 0; i < 2; i++) {
        uint32_t tid = mCurrentTextureIds[i];
        if (tid < mTextures.size()) {
            pc.textureWidth[i] = (int32_t)mTextures[tid].width;
            pc.textureHeight[i] = (int32_t)mTextures[tid].height;
            pc.textureFiltering[i] = mTextures[tid].linearFiltering ? FILTER_THREE_POINT : FILTER_LINEAR;
        }
    }
    vkCmdPushConstants(cmd, mCurrentShader->pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    // Bind descriptor set with textures
    bool needsDescriptor = false;
    for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
        if (mCurrentShader->usedTextures[i]) { needsDescriptor = true; break; }
    }

    if (needsDescriptor && mCurrentShader->descriptorSetLayout != VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = mFrameDescPool[mCurrentFrame];
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &mCurrentShader->descriptorSetLayout;
        VkDescriptorSet dset;
        VK_CHECK(vkAllocateDescriptorSets(mDevice, &dsai, &dset));

        // Write descriptors
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorImageInfo> imageInfos(SHADER_MAX_TEXTURES);
        uint32_t bindings[SHADER_MAX_TEXTURES] = {0,1,2,3,4,5};
        for (int i = 0; i < SHADER_MAX_TEXTURES; i++) {
            if (!mCurrentShader->usedTextures[i]) continue;
            uint32_t tid = mCurrentTextureIds[i];
            if (tid >= mTextures.size()) continue;
            auto& tex = mTextures[tid];
            if (!tex.view || !tex.sampler) continue;

            imageInfos[i].sampler = tex.sampler;
            imageInfos[i].imageView = tex.view;
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = dset;
            w.dstBinding = bindings[i];
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &imageInfos[i];
            writes.push_back(w);
        }
        if (!writes.empty())
            vkUpdateDescriptorSets(mDevice, (uint32_t)writes.size(), writes.data(), 0, nullptr);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mCurrentShader->pipelineLayout,
                                0, 1, &dset, 0, nullptr);
    }

    // Upload vertex data
    size_t dataSize = bufVboLen * sizeof(float);
    if (dataSize > VERTEX_BUFFER_SIZE) {
        SPDLOG_ERROR("Vertex buffer overflow: {} > {}", dataSize, VERTEX_BUFFER_SIZE);
        return;
    }
    memcpy(mVertexBufferMapped, bufVbo, dataSize);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mVertexBuffer, &offset);
    vkCmdDraw(cmd, (uint32_t)(bufVboNumTris * 3), 1, 0, 0);
}

// ---- Pixel depth query -------------------------------------------------------

std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIVulkan::GetPixelDepth(int fbId, const std::set<std::pair<float, float>>& coords) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> result;
    if (fbId >= (int)mFrameBuffers.size() || fbId == 0) return result;
    auto& fb = mFrameBuffers[fbId];
    if (!fb.hasDepth || !fb.depthImage) return result;

    // Read depth buffer as R32_SFLOAT
    VkDeviceSize size = (VkDeviceSize)fb.width * fb.height * sizeof(float);
    VkBuffer stageBuf;
    VmaAllocation stageAlloc;
    VmaAllocationInfo stageInfo{};
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo vaci{};
    vaci.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    vaci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VK_CHECK(vmaCreateBuffer(mAllocator, &bci, &vaci, &stageBuf, &stageAlloc, &stageInfo));

    VkCommandBuffer cmd = BeginOneTimeSubmit();
    TransitionImageLayout(cmd, fb.depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
    region.imageExtent = { fb.width, fb.height, 1 };
    vkCmdCopyImageToBuffer(cmd, fb.depthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stageBuf, 1, &region);
    TransitionImageLayout(cmd, fb.depthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    EndOneTimeSubmit(cmd);

    const float* depthData = (const float*)stageInfo.pMappedData;
    for (auto& [nx, ny] : coords) {
        int px = (int)(nx * fb.width);
        int py = (int)(ny * fb.height);
        px = std::clamp(px, 0, (int)fb.width - 1);
        py = std::clamp(py, 0, (int)fb.height - 1);
        float d = depthData[py * fb.width + px];
        result[{nx, ny}] = (uint16_t)(d * 65535.0f);
    }
    vmaDestroyBuffer(mAllocator, stageBuf, stageAlloc);
    return result;
}

// ---- Misc -------------------------------------------------------------------

const char* GfxRenderingAPIVulkan::GetName() { return "Vulkan"; }
int GfxRenderingAPIVulkan::GetMaxTextureSize() { return 16384; }

GfxClipParameters GfxRenderingAPIVulkan::GetClipParameters() {
    // Vulkan clip space: Z in [0,1], Y inverted relative to OpenGL
    bool invertY = false;
    if (mCurrentFbId < (int)mFrameBuffers.size())
        invertY = mFrameBuffers[mCurrentFbId].invertY;
    return { true, invertY };
}

void GfxRenderingAPIVulkan::SetTextureFilter(FilteringMode mode) { mFilterMode = mode; }
FilteringMode GfxRenderingAPIVulkan::GetTextureFilter() { return mFilterMode; }

void GfxRenderingAPIVulkan::SetSrgbMode() {
    mSrgbMode = true;
}

ImTextureID GfxRenderingAPIVulkan::GetTextureById(int id) {
    if ((uint32_t)id >= mTextures.size()) return nullptr;
    auto& tex = mTextures[id];
    if (!tex.view || !tex.sampler) return nullptr;
    return ImGui_ImplVulkan_AddTexture(tex.sampler, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void* GfxRenderingAPIVulkan::GetFramebufferTextureId(int fbId) {
    if (fbId == 0 || fbId >= (int)mFrameBuffers.size()) return nullptr;
    auto& fb = mFrameBuffers[fbId];
    VkImageView view = (fb.msaaLevel > 1 && fb.colorResolvView) ? fb.colorResolvView : fb.colorView;
    if (!view) return nullptr;
    VkSampler sampler = VK_NULL_HANDLE;
    // Create a default sampler for fb reads if needed
    if (mTextures.size() > (uint32_t)fbId + 0x4000) {
        sampler = mTextures[fbId + 0x4000].sampler;
    }
    if (!sampler) {
        VulkanTexture tmp{};
        sampler = GetOrCreateSampler(tmp);
        // leak is acceptable as this is a rare path
    }
    return ImGui_ImplVulkan_AddTexture(sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// ---- ImGui integration ------------------------------------------------------

void GfxRenderingAPIVulkan::VulkanGuiInit() {
    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = mInstance;
    ii.PhysicalDevice = mPhysDevice;
    ii.Device = mDevice;
    ii.QueueFamily = mGraphicsQueueFamily;
    ii.Queue = mGraphicsQueue;
    ii.DescriptorPool = mDescriptorPool;
    ii.MinImageCount = 2;
    ii.ImageCount = (uint32_t)mSwapchainImages.size();
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = true;
    ii.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ii.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineRenderingCreateInfo.pColorAttachmentFormats = &mSwapchainFormat;
    ImGui_ImplVulkan_Init(&ii);
}

void GfxRenderingAPIVulkan::VulkanRenderDrawData(ImDrawData* data) {
    if (!mFrameStarted) return;
    // ImGui renders into the currently active command buffer
    ImGui_ImplVulkan_RenderDrawData(data, mFrames[mCurrentFrame].cmdBuf);
}

// ---- Utility ----------------------------------------------------------------

VkCommandBuffer GfxRenderingAPIVulkan::BeginOneTimeSubmit() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = mTransferPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(mDevice, &ai, &cmd));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void GfxRenderingAPIVulkan::EndOneTimeSubmit(VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkFence fence;
    VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(mDevice, &fi, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(mDevice, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(mDevice, fence, nullptr);
    vkFreeCommandBuffers(mDevice, mTransferPool, 1, &cmd);
}

void GfxRenderingAPIVulkan::TransitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                                   VkImageLayout oldLayout, VkImageLayout newLayout,
                                                   VkImageAspectFlags aspect) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { aspect, 0, 1, 0, 1 };

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

    auto stageAndAccess = [](VkImageLayout layout,
                              VkPipelineStageFlags& stage, VkAccessFlags& access) {
        switch (layout) {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; access = 0; break;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                stage = VK_PIPELINE_STAGE_TRANSFER_BIT; access = VK_ACCESS_TRANSFER_READ_BIT; break;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                stage = VK_PIPELINE_STAGE_TRANSFER_BIT; access = VK_ACCESS_TRANSFER_WRITE_BIT; break;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; access = VK_ACCESS_SHADER_READ_BIT; break;
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; break;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
                access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; break;
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; access = 0; break;
            default:
                stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT; break;
        }
    };

    stageAndAccess(oldLayout, srcStage, barrier.srcAccessMask);
    stageAndAccess(newLayout, dstStage, barrier.dstAccessMask);

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace Fast
#endif // ENABLE_VULKAN
