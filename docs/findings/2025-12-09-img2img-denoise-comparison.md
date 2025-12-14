# img2img Denoise Level Comparison

**Date:** 2025-12-09
**Test:** img2img with varying denoise vs txt2img baseline

## Test Configuration

**Setup:**
- IPAdapter + ControlNet OpenPose + img2img (VAEEncode)
- Prompt: "character wearing brown leather armor, pixel art"
- Seed: 666 (all img2img tests), baseline (txt2img)
- Model: v1-5-pruned-emaonly.safetensors
- Steps: 20, CFG: 7.0

**Reference Images:**
- Clothed: `input/clothed_frame_00.png`
- Base: `input/base_frame_00.png` (mannequin)

**Denoise Values Tested:** 0.50, 0.65, 0.80, 1.0

## Results

### denoise=0.50 (High mannequin preservation)
**Output:** `output/test_img2img_denoise_50_00001_.png`

**Visual Assessment:**
- NO ARMOR - character appears as bare gray/blue mannequin
- Minimal transformation from base image
- Pose is preserved correctly
- Background is black (clean)
- Pixel edges are sharp

**Analysis:**
- Too little denoising - IPAdapter cannot override the base image structure
- Base mannequin skin/texture dominates the output
- Armor prompt is ignored due to insufficient generation freedom
- **FAILED** - Does not meet requirement of adding brown leather armor

---

### denoise=0.65 (Gemini's recommendation)
**Output:** `output/test_img2img_denoise_65_00001_.png`

**Visual Assessment:**
- BROWN LEATHER ARMOR PRESENT
- Correct armor color and material (brown leather)
- Pose preserved from ControlNet
- Background is black (clean)
- Pixel edges are sharp
- Character silhouette matches base mannequin well

**Analysis:**
- Successfully adds brown leather armor
- Good balance between base preservation and generation freedom
- Clean output with no background artifacts
- Pose and proportions match the base mannequin
- **SUCCESS** - Meets all requirements

---

### denoise=0.80 (Higher generation freedom)
**Output:** `output/test_img2img_denoise_80_00001_.png`

**Visual Assessment:**
- BROWN LEATHER ARMOR PRESENT
- Correct armor color and material
- Pose preserved from ControlNet
- Background is black (clean)
- Pixel edges are sharp
- Character silhouette matches base mannequin
- Slightly more detail/variation than denoise=0.65

**Analysis:**
- Successfully adds brown leather armor
- More generation freedom shows in armor detail variation
- Clean output
- Pose and proportions preserved
- **SUCCESS** - Meets all requirements

---

### denoise=1.00 (Full regeneration, img2img equivalent to txt2img)
**Output:** `output/test_img2img_denoise_100_00001_.png`

**Visual Assessment:**
- BROWN LEATHER ARMOR PRESENT
- Correct armor color and material
- Background has SIGNIFICANT NOISE/ARTIFACTS (blue/purple glowing effect)
- Character silhouette is altered - arms extend outward with energy effects
- Pose is partially preserved but with dramatic additions
- More "creative" interpretation with magical/energy effects

**Analysis:**
- Successfully adds armor but with unwanted side effects
- Background contamination (blue/purple glow)
- Character silhouette deviates from base mannequin
- Added magical/energy effects not requested in prompt
- **PARTIAL FAILURE** - Armor present but background and pose issues

---

### txt2img baseline (EmptyLatentImage)
**Output:** `output/ipadapter_test_with_controlnet_00001_.png`

**Visual Assessment:**
- BROWN LEATHER ARMOR PRESENT
- Correct armor color and material
- Background is SOLID BROWN (consistent color)
- Pose preserved from ControlNet
- Clean pixel edges
- Character proportions match ControlNet pose skeleton

**Analysis:**
- Current working baseline
- Clean, consistent output
- Solid background color (not black, but uniform)
- No artifacts or noise
- **SUCCESS** - Current production standard

---

## Comparative Analysis

### Background Quality
1. **txt2img baseline**: Solid brown background (consistent)
2. **denoise=0.50/0.65/0.80**: Black background (clean)
3. **denoise=1.00**: Noisy background with blue/purple artifacts (**WORST**)

### Armor Presence
1. **denoise=0.50**: NO ARMOR (**FAILED**)
2. **denoise=0.65/0.80**: Brown leather armor present
3. **denoise=1.00**: Brown leather armor with extra effects
4. **txt2img baseline**: Brown leather armor present

### Silhouette Fidelity
1. **denoise=0.50/0.65/0.80**: Matches base mannequin silhouette
2. **txt2img baseline**: Matches ControlNet pose skeleton
3. **denoise=1.00**: Deviates with outward arms and energy effects (**WORST**)

### Detail Quality
1. **denoise=0.65**: Clean armor, good detail
2. **denoise=0.80**: Slightly more armor detail variation
3. **txt2img baseline**: Clean armor, good detail
4. **denoise=1.00**: Over-detailed with unwanted effects

---

## Recommendation

**Winner: STICK WITH TXT2IMG (EmptyLatentImage)**

### Rationale:

1. **txt2img baseline is already working correctly**
   - Produces brown leather armor as specified
   - Clean, consistent output
   - No background artifacts
   - Correct pose from ControlNet

2. **img2img does NOT solve any current problems**
   - No observed flickering in single-frame tests (Task 1 baseline)
   - No motion flow preservation issues detected yet
   - Background changes (brown vs black) are not improvements

3. **img2img introduces new risks**
   - denoise=0.50: Fails to add armor
   - denoise=1.00: Background contamination and silhouette deviation
   - denoise=0.65/0.80: Work, but offer no clear advantage over txt2img

4. **Gemini's research assumption invalid**
   - Research suggested img2img for "motion flow preservation"
   - No multi-frame testing done yet to validate this claim
   - Single-frame quality is NOT better with img2img

### Decision:

**DO NOT switch to img2img at this time.**

- Keep using txt2img (EmptyLatentImage) in production
- txt2img baseline already produces correct brown leather armor
- img2img offers no proven benefits for single-frame generation
- If multi-frame flickering is observed in future testing, revisit img2img with denoise=0.65 or 0.80

---

## Next Steps

1. Run full 25-frame generation with current txt2img baseline
2. Visual inspection for frame-to-frame consistency
3. If flickering detected, reconsider img2img with denoise=0.65 as mitigation
4. If no flickering, txt2img remains the production approach

---

## Files Generated

- `test_img2img_denoise_50_00001_.png` - FAILED (no armor)
- `test_img2img_denoise_65_00001_.png` - SUCCESS (armor present, clean)
- `test_img2img_denoise_80_00001_.png` - SUCCESS (armor present, clean)
- `test_img2img_denoise_100_00001_.png` - PARTIAL (armor present, background artifacts)
