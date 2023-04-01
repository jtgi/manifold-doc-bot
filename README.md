### manny
manifold doc embeddings + gpt + q&a bot.

example
```sh
curl -s https://askmanny.fly.dev/question \
-H 'content-type: application/json' \
-d '{ "query": "why isnt my token loading?" }' \
> | python -m json.tool
{
    "answer": "### Question: \n  why isnt my token loading?\n  ### Answer: \n  It could be due to various reasons depending on the platform you are trying to load the token on. If your token isn't showing up on Opensea, it might take a few hours for Opensea to ingest the token data. If it's still not showing up, you can contact Opensea's support team. If your token isn't loading on Foundation, please make sure that the file size/type is correct as Foundation only displays images/videos under 50mb. Additionally, if the issue is with the token displaying on all platforms, there may have been an issue during the minting process. You can try double-checking metadata and compatibility, and if the issue persists, you can contact the platform's support team for assistance. \n  ### Sources: \n  https://docs.manifold.xyz/v/manifold-studio/essentials/solve-platform-display-issues, https://docs.manifold.xyz/v/manifold-studio/essentials/faq\n  ### All relevant sources:\n  https://docs.manifold.xyz/v/manifold-studio/essentials/faq https://docs.manifold.xyz/v/manifold-studio/essentials/solve-platform-display-issues\n  "
}

`flyctl deploy`
