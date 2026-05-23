# Setup Gemini API Key

## Get Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Get API key"  
3. Copy your API key

## Add to Environment

Edit `.env.local`:

```bash
GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
```

Then restart the backend:

```bash
pkill -f "python.*atlantean_backend"
python atlantean_backend.py
```

## Or Use Temporary Key (For Testing)

You can also pass the API key directly in your first message by storing it in localStorage:

Open browser console (F12) and run:

```javascript
localStorage.setItem('gemini_api_key', 'YOUR_API_KEY');
```

Then refresh the page.

## Verify It's Working

When you chat, you should see:
- ✅ Real Gemini responses (not "This is a demo response")
- ✅ Intelligence state context in responses
- ✅ Learning signals being applied

## Current Status

Without API key: You'll see demo responses with intelligence field stats

With API key: Full Gemini AI responses powered by Atlantean Intelligence Core!
