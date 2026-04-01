# Quick Start - Stable Demo Version

## Run Application
```bash
streamlit run app.py
```

## Key Features
✅ **Never crashes** - All imports wrapped safely
✅ **Always shows UI** - All 8 tabs render
✅ **Fallback modes** - Works even if AI fails
✅ **Stable FPS** - 20-30 FPS on most hardware
✅ **Simple tracking** - Works with or without DeepSORT

## System Status Indicators
- ✅ = Fully working
- ✅ (fallback) = Working with basic mode
- ❌ = Not available (app still works)

## What Works
1. **Live Monitoring** - Camera feed + detection
2. **Analytics** - Basic statistics
3. **Person Intelligence** - Tracking info
4. **Image Intelligence** - Upload analysis
5. **AI Intelligence** - Gemini analysis (with fallback)
6. **Insights** - Activity logs
7. **Crowd Monitoring** - Person counting
8. **Use Cases** - Documentation

## Fallback Behaviors
- **No DeepFace**: Shows "Unknown" for age/gender/emotion
- **No Gemini**: Returns rule-based analysis text
- **No DeepSORT**: Uses simple bounding box tracker
- **No GPU**: Runs on CPU automatically

## Performance Tips
- App processes every 2nd frame for speed
- Frame resized to 640x480
- GPU used automatically if available
- All heavy operations cached

## Troubleshooting
**Camera won't start**: Check permissions, close other apps using camera
**Low FPS**: Normal on CPU, should be 20-30 FPS on GPU
**Module errors**: App will continue with fallback modes

This version prioritizes **STABILITY** over perfect AI accuracy.
