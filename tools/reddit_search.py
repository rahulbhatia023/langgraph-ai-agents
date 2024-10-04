from typing import Union, Dict

import praw
from langchain_core.tools import BaseTool
from praw.models import Comment
from pydantic import Field, BaseModel


class Rec(BaseModel):
    title: str
    description: str
    comments: list[str]

    def __str__(self):
        return f"\n\n**Title**:\n\n{self.title}\n\n\n**Description**:\n\n{self.description}\n\n\n**Comments:**\n\n{'\n\n'.join(self.comments)}\n\n ---------------------------"


class RedditSearchTool(BaseTool):
    reddit_client_id: str = Field(..., description="Reddit API client ID")
    reddit_client_secret: str = Field(..., description="Reddit API client secret")
    reddit_user_agent: str = Field(..., description="Reddit API user agent")

    name: str = "reddit-search"
    description: str = "Provides access to search reddit"

    def _run(self, query: str) -> Union[Dict, str]:
        reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            user_agent=self.reddit_user_agent,
        )

        results = reddit.subreddit("all").search(query)

        recs = []
        for submission in results:
            title = submission.title
            description = submission.selftext
            comments = []
            for comment in submission.comments.list():
                if isinstance(comment, Comment) and comment.ups >= 10:
                    author = comment.author.name if comment.author else "unknown"
                    comments.append(
                        f"{author} (upvotes: {comment.ups}): {comment.body}"
                    )
            comments = comments[:3]
            if len(comments) == 3:
                recs.append(
                    Rec(title=title, description=description, comments=comments)
                )
            if len(recs) == 3:
                break

        recommendations = ""
        for rec in recs:
            recommendations += str(rec) + "\n\n"

        return recommendations
