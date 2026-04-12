#ifdef ENABLE_VULKAN
#pragma once

#include "gfx_rendering_api.h"
#include "gfx_sdl.h"
#include "fast/interpreter.h"

#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <shaderc/shaderc.hpp>

#include <map>
#include <vector>
#include <unordered_map>
#include <set>
#include <string>

struct CCFeatures;

namespace Fast {

static constexpr int VULKAN_FRAMES_IN_FLIGHT = 2;

struct ShaderProgramVulkan {
    // Two pipelines: [0] = blend disabled, [1] = blend enabled
    VkPipeline pipelines[2];
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    uint8_t numInputs;
    bool usedTextures[SHADER_MAX_TEXTURES];
    uint8_t numFloats;
    uint8_t numAttribs;
    uint8_t attribSizes[16];
};

struct VulkanTexture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFilter minFilter = VK_FILTER_NEAREST;
    VkFilter magFilter = VK_FILTER_NEAREST;
    VkSamplerAddressMode wrapS = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode wrapT = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    bool linearFiltering = false;
};

struct VulkanFramebuffer {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t msaaLevel = 1;
    bool hasDepth = false;
    bool invertY = false;

    VkImage colorImage = VK_NULL_HANDLE;
    VmaAllocation colorAlloc = VK_NULL_HANDLE;
    VkImageView colorView = VK_NULL_HANDLE;
    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;

    // MSAA resolve target (non-null only when msaaLevel > 1)
    VkImage colorResolvImage = VK_NULL_HANDLE;
    VmaAllocation colorResolvAlloc = VK_NULL_HANDLE;
    VkImageView colorResolvView = VK_NULL_HANDLE;

    VkImage depthImage = VK_NULL_HANDLE;
    VmaAllocation depthAlloc = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;
};

struct VulkanFrameData {
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    VkSemaphore imageAvailableSem = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSem = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
};

struct PushConstants {
    int32_t frameCount;
    float noiseScale;
    int32_t textureWidth[2];
    int32_t textureHeight[2];
    int32_t textureFiltering[2];
};
static_assert(sizeof(PushConstants) <= 128, "Push constants too large");

class GfxRenderingAPIVulkan final : public GfxRenderingAPI {
  public:
    explicit GfxRenderingAPIVulkan(GfxWindowBackendSDL2* sdlBackend);
    ~GfxRenderingAPIVulkan() override;

    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linearFilter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depthTest, bool zUpd) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float bufVbo[], size_t bufVboLen, size_t bufVboNumTris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fbId, uint32_t width, uint32_t height, uint32_t msaaLevel, bool oglInvertY,
                                     bool renderTarget, bool hasDepthBuffer, bool canExtractDepth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarget, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fbId, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;

    // Called by Gui.cpp for ImGui Vulkan backend setup/teardown
    void VulkanGuiInit();
    void VulkanRenderDrawData(ImDrawData* data);

  private:
    GfxWindowBackendSDL2* mSdlBackend;

    // Core Vulkan
    VkInstance mInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysDevice = VK_NULL_HANDLE;
    VkDevice mDevice = VK_NULL_HANDLE;
    VkQueue mGraphicsQueue = VK_NULL_HANDLE;
    VkQueue mPresentQueue = VK_NULL_HANDLE;
    uint32_t mGraphicsQueueFamily = UINT32_MAX;
    uint32_t mPresentQueueFamily = UINT32_MAX;
    VkSurfaceKHR mSurface = VK_NULL_HANDLE;
    VmaAllocator mAllocator = VK_NULL_HANDLE;

    // Swapchain
    VkSwapchainKHR mSwapchain = VK_NULL_HANDLE;
    VkFormat mSwapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D mSwapchainExtent = {};
    std::vector<VkImage> mSwapchainImages;
    std::vector<VkImageView> mSwapchainImageViews;
    uint32_t mSwapchainImageIndex = 0;
    bool mVsyncEnabled = true;

    // Frames in flight
    VulkanFrameData mFrames[VULKAN_FRAMES_IN_FLIGHT];
    int mCurrentFrame = 0;
    bool mFrameStarted = false;

    // One-time submit pool
    VkCommandPool mTransferPool = VK_NULL_HANDLE;

    // Vertex buffer (persistently mapped staging, device-local)
    static constexpr size_t VERTEX_BUFFER_SIZE = 256 * 1024 * sizeof(float);
    VkBuffer mVertexBuffer = VK_NULL_HANDLE;
    VmaAllocation mVertexBufferAlloc = VK_NULL_HANDLE;
    void* mVertexBufferMapped = nullptr;

    // Pipeline cache
    VkPipelineCache mPipelineCache = VK_NULL_HANDLE;

    // Shader program pool keyed by (shaderId0, shaderId1)
    std::map<std::pair<uint64_t, uint64_t>, ShaderProgramVulkan> mShaderPool;

    // Textures
    std::vector<VulkanTexture> mTextures;
    uint32_t mCurrentTextureIds[SHADER_MAX_TEXTURES] = {};
    uint32_t mCurrentTile = 0;

    // Framebuffers
    std::vector<VulkanFramebuffer> mFrameBuffers;
    int mCurrentFbId = 0;

    // Draw state
    ShaderProgramVulkan* mCurrentShader = nullptr;
    bool mBlendEnabled = false;
    bool mDepthTestEnabled = false;
    bool mDepthWriteEnabled = false;
    bool mZmodeDecal = false;
    float mCurrentNoiseScale = 1.0f;
    uint32_t mFrameCount = 0;
    FilteringMode mFilterMode = FILTER_THREE_POINT;

    // Descriptor pool for ImGui and textures
    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;

    // shaderc compiler (persistent instance)
    shaderc::Compiler mShadercCompiler;
    shaderc::CompileOptions mShadercOptions;

    // Validation layer support
#ifdef _DEBUG
    VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;
#endif

    // Helper methods
    void CreateInstance();
    void CreateSurface();
    void SelectPhysicalDevice();
    void CreateDevice();
    void CreateAllocator();
    void CreateCommandPools();
    void CreateSyncObjects();
    void CreateVertexBuffer();
    void CreatePipelineCache();
    void CreateDescriptorPool();
    void CreateSwapchain();
    void DestroySwapchain();

    VkCommandBuffer BeginOneTimeSubmit();
    void EndOneTimeSubmit(VkCommandBuffer cmd);

    void TransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                               VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    VkShaderModule CompileShader(const std::string& source, shaderc_shader_kind kind, const char* name);

    std::string BuildVertexShader(const CCFeatures& cc_features, uint8_t& numFloatsOut, uint8_t& numAttribsOut,
                                  uint8_t attribSizes[16]);
    std::string BuildFragmentShader(const CCFeatures& cc_features);

    ShaderProgramVulkan* BuildShaderProgram(uint64_t shaderId0, uint64_t shaderId1);

    void RecreateFramebufferImages(VulkanFramebuffer& fb);
    void DestroyFramebufferImages(VulkanFramebuffer& fb);

    VkSampler GetOrCreateSampler(const VulkanTexture& tex);

    VkDescriptorSet AllocateDescriptorSet(VkDescriptorSetLayout layout);
    void FreeDescriptorSets();

    // Per-frame descriptor set cache (reset each frame)
    std::vector<VkDescriptorSet> mFrameDescSets[VULKAN_FRAMES_IN_FLIGHT];
    size_t mFrameDescSetIdx[VULKAN_FRAMES_IN_FLIGHT] = {};

    VkDescriptorPool mFrameDescPool[VULKAN_FRAMES_IN_FLIGHT] = {};
};

} // namespace Fast
#endif // ENABLE_VULKAN
