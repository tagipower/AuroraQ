#!/usr/bin/env python3
"""
API 키 등록 후 실제 연동 검증
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import aiohttp
import json

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from SharedCore.data_collection.news_aggregation_system import AuroraQNewsAggregator
from SharedCore.sentiment_engine.news_collectors.news_collector import NewsCollector

class APIKeyValidator:
    """API 키 유효성 및 연동 검증"""
    
    def __init__(self):
        self.results = {
            "validation_time": datetime.now().isoformat(),
            "api_tests": {},
            "system_tests": {},
            "overall_status": "unknown"
        }
    
    def log_result(self, test_name: str, status: str, message: str, data: dict = None):
        """테스트 결과 로깅"""
        result = {
            "status": status,  # pass, fail, warning
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        self.results["api_tests"][test_name] = result
        
        # 콘솔 출력
        status_emoji = {"pass": "✅", "fail": "❌", "warning": "⚠️"}
        print(f"{status_emoji.get(status, '❓')} {test_name}: {message}")
        
        if data and status != "pass":
            print(f"   Details: {data}")
    
    async def test_newsapi_connection(self):
        """NewsAPI 연결 테스트"""
        api_key = os.getenv("NEWSAPI_KEY")
        
        if not api_key or api_key == "your_newsapi_key_here":
            self.log_result("NewsAPI Connection", "warning", "API key not configured")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}&pageSize=5"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        article_count = len(data.get('articles', []))
                        self.log_result(
                            "NewsAPI Connection", 
                            "pass", 
                            f"Successfully connected, fetched {article_count} articles",
                            {"status_code": response.status, "articles": article_count}
                        )
                        return True
                    else:
                        error_data = await response.text()
                        self.log_result(
                            "NewsAPI Connection", 
                            "fail", 
                            f"API returned status {response.status}",
                            {"status_code": response.status, "response": error_data[:200]}
                        )
                        return False
        except Exception as e:
            self.log_result("NewsAPI Connection", "fail", f"Connection failed: {str(e)}")
            return False
    
    async def test_finnhub_connection(self):
        """Finnhub 연결 테스트"""
        api_key = os.getenv("FINNHUB_API_KEY")
        
        if not api_key or api_key == "your_finnhub_key_here":
            self.log_result("Finnhub Connection", "warning", "API key not configured")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'c' in data and data['c'] is not None:  # current price
                            self.log_result(
                                "Finnhub Connection", 
                                "pass", 
                                f"Successfully connected, AAPL price: ${data['c']}",
                                {"status_code": response.status, "sample_data": data}
                            )
                            return True
                        else:
                            self.log_result(
                                "Finnhub Connection", 
                                "fail", 
                                "Invalid response format",
                                {"response": data}
                            )
                            return False
                    else:
                        error_data = await response.text()
                        self.log_result(
                            "Finnhub Connection", 
                            "fail", 
                            f"API returned status {response.status}",
                            {"status_code": response.status, "response": error_data[:200]}
                        )
                        return False
        except Exception as e:
            self.log_result("Finnhub Connection", "fail", f"Connection failed: {str(e)}")
            return False
    
    async def test_reddit_connection(self):
        """Reddit API 연결 테스트"""
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "AuroraQ News Collector")
        
        if not client_id or not client_secret or client_id == "your_reddit_client_id":
            self.log_result("Reddit Connection", "warning", "API credentials not configured")
            return False
        
        try:
            # Reddit OAuth 인증
            import base64
            auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            
            async with aiohttp.ClientSession() as session:
                # 토큰 요청
                token_url = "https://www.reddit.com/api/v1/access_token"
                token_headers = {
                    "Authorization": f"Basic {auth}",
                    "User-Agent": user_agent
                }
                token_data = {
                    "grant_type": "client_credentials"
                }
                
                async with session.post(token_url, headers=token_headers, data=token_data, timeout=10) as response:
                    if response.status == 200:
                        token_result = await response.json()
                        access_token = token_result.get("access_token")
                        
                        if access_token:
                            # 실제 API 호출 테스트
                            api_headers = {
                                "Authorization": f"bearer {access_token}",
                                "User-Agent": user_agent
                            }
                            api_url = "https://oauth.reddit.com/r/cryptocurrency/hot?limit=3"
                            
                            async with session.get(api_url, headers=api_headers, timeout=10) as api_response:
                                if api_response.status == 200:
                                    api_data = await api_response.json()
                                    post_count = len(api_data.get('data', {}).get('children', []))
                                    self.log_result(
                                        "Reddit Connection", 
                                        "pass", 
                                        f"Successfully connected, fetched {post_count} posts from r/cryptocurrency",
                                        {"status_code": api_response.status, "posts": post_count}
                                    )
                                    return True
                                else:
                                    error_data = await api_response.text()
                                    self.log_result(
                                        "Reddit Connection", 
                                        "fail", 
                                        f"API call failed with status {api_response.status}",
                                        {"status_code": api_response.status, "response": error_data[:200]}
                                    )
                                    return False
                        else:
                            self.log_result("Reddit Connection", "fail", "Failed to get access token")
                            return False
                    else:
                        error_data = await response.text()
                        self.log_result(
                            "Reddit Connection", 
                            "fail", 
                            f"Token request failed with status {response.status}",
                            {"status_code": response.status, "response": error_data[:200]}
                        )
                        return False
        except Exception as e:
            self.log_result("Reddit Connection", "fail", f"Connection failed: {str(e)}")
            return False
    
    async def test_news_aggregation_system(self):
        """통합 뉴스 시스템 테스트"""
        try:
            aggregator = AuroraQNewsAggregator()
            
            # 시스템 상태 확인
            health = await aggregator.get_system_health()
            
            active_collectors = health.get('active_collectors', 0)
            total_collectors = health.get('total_collectors', 0)
            
            if health.get('status') == 'healthy':
                self.log_result(
                    "News Aggregation System", 
                    "pass", 
                    f"System healthy with {active_collectors}/{total_collectors} collectors active",
                    {"health_data": health}
                )
            elif health.get('status') == 'degraded':
                self.log_result(
                    "News Aggregation System", 
                    "warning", 
                    f"System degraded: {active_collectors}/{total_collectors} collectors active"
                )
            else:
                self.log_result(
                    "News Aggregation System", 
                    "fail", 
                    f"System unhealthy: {health.get('status')}"
                )
            
            await aggregator.close_all()
            return active_collectors > 0
            
        except Exception as e:
            self.log_result("News Aggregation System", "fail", f"System test failed: {str(e)}")
            return False
    
    async def test_legacy_compatibility(self):
        """기존 인터페이스 호환성 테스트"""
        try:
            collector = NewsCollector()
            await collector.connect()
            
            # 암호화폐 뉴스 수집 테스트
            crypto_news = await collector.get_latest_crypto_news(hours_back=6, max_articles=10)
            
            if len(crypto_news) > 0:
                # 감정 분석 테스트
                sentiment_summary = await collector.get_sentiment_summary(crypto_news)
                
                self.log_result(
                    "Legacy Interface Compatibility", 
                    "pass", 
                    f"Legacy interface works: {len(crypto_news)} articles, sentiment: {sentiment_summary['overall_sentiment']:.2f}",
                    {"article_count": len(crypto_news), "sentiment_data": sentiment_summary}
                )
            else:
                self.log_result(
                    "Legacy Interface Compatibility", 
                    "warning", 
                    "Legacy interface works but no articles collected"
                )
            
            await collector.close()
            return True
            
        except Exception as e:
            self.log_result("Legacy Interface Compatibility", "fail", f"Legacy test failed: {str(e)}")
            return False
    
    async def test_telegram_bot(self):
        """Telegram 봇 연결 테스트"""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or bot_token == "your_telegram_bot_token":
            self.log_result("Telegram Bot", "warning", "Bot token not configured")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                # 봇 정보 확인
                url = f"https://api.telegram.org/bot{bot_token}/getMe"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            bot_info = data.get('result', {})
                            bot_name = bot_info.get('username', 'Unknown')
                            self.log_result(
                                "Telegram Bot", 
                                "pass", 
                                f"Bot @{bot_name} is active and accessible",
                                {"bot_info": bot_info}
                            )
                            
                            # 테스트 메시지 전송 시도 (Chat ID가 있을 경우)
                            if chat_id and chat_id != "your_telegram_chat_id":
                                test_message = f"🧪 AuroraQ 연동 테스트 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                                send_data = {
                                    "chat_id": chat_id,
                                    "text": test_message
                                }
                                
                                async with session.post(send_url, json=send_data, timeout=10) as send_response:
                                    if send_response.status == 200:
                                        send_result = await send_response.json()
                                        if send_result.get('ok'):
                                            self.log_result(
                                                "Telegram Message Test", 
                                                "pass", 
                                                "Test message sent successfully"
                                            )
                                        else:
                                            self.log_result(
                                                "Telegram Message Test", 
                                                "warning", 
                                                f"Message send failed: {send_result.get('description')}"
                                            )
                            
                            return True
                        else:
                            self.log_result("Telegram Bot", "fail", f"Bot API error: {data.get('description')}")
                            return False
                    else:
                        error_data = await response.text()
                        self.log_result(
                            "Telegram Bot", 
                            "fail", 
                            f"Bot API returned status {response.status}",
                            {"status_code": response.status, "response": error_data[:200]}
                        )
                        return False
        except Exception as e:
            self.log_result("Telegram Bot", "fail", f"Bot test failed: {str(e)}")
            return False
    
    async def run_validation(self):
        """전체 검증 실행"""
        print("🧪 AuroraQ v2.0 API 키 및 연동 검증 시작")
        print("=" * 60)
        
        # API 키 환경 변수 확인
        print("📋 등록된 API 키 확인:")
        api_keys = {
            "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
            "FINNHUB_API_KEY": os.getenv("FINNHUB_API_KEY", ""),
            "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID", ""),
            "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET", ""),
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "")
        }
        
        configured_keys = 0
        for key, value in api_keys.items():
            if value and value != "" and "your_" not in value:
                print(f"   ✅ {key}: 설정됨")
                configured_keys += 1
            else:
                print(f"   ❌ {key}: 미설정")
        
        print(f"\n📊 API 키 등록률: {configured_keys}/{len(api_keys)} ({configured_keys/len(api_keys)*100:.1f}%)")
        
        # 개별 API 연결 테스트
        print(f"\n🔗 API 연결 테스트:")
        api_tests = [
            self.test_newsapi_connection(),
            self.test_finnhub_connection(),
            self.test_reddit_connection(),
            self.test_telegram_bot()
        ]
        
        api_results = await asyncio.gather(*api_tests, return_exceptions=True)
        successful_apis = sum(1 for result in api_results if result is True)
        
        # 시스템 통합 테스트
        print(f"\n🏗️ 시스템 통합 테스트:")
        system_tests = [
            self.test_news_aggregation_system(),
            self.test_legacy_compatibility()
        ]
        
        system_results = await asyncio.gather(*system_tests, return_exceptions=True)
        successful_systems = sum(1 for result in system_results if result is True)
        
        # 최종 결과 요약
        print(f"\n" + "=" * 60)
        print(f"📊 검증 결과 요약")
        print(f"=" * 60)
        
        total_tests = len(api_tests) + len(system_tests)
        total_successful = successful_apis + successful_systems
        
        print(f"API 연결 테스트: {successful_apis}/{len(api_tests)} 성공")
        print(f"시스템 테스트: {successful_systems}/{len(system_tests)} 성공")
        print(f"전체 성공률: {total_successful}/{total_tests} ({total_successful/total_tests*100:.1f}%)")
        
        # 전체 평가
        if total_successful == total_tests:
            status = "EXCELLENT"
            print(f"\n🎉 완벽! 모든 API와 시스템이 정상 작동합니다!")
        elif total_successful >= total_tests * 0.8:
            status = "GOOD"
            print(f"\n✅ 우수! 대부분의 기능이 정상 작동합니다.")
        elif total_successful >= total_tests * 0.5:
            status = "FAIR"
            print(f"\n⚠️ 보통. 일부 기능에 문제가 있지만 기본 동작 가능합니다.")
        else:
            status = "POOR"
            print(f"\n❌ 문제 발생. 주요 기능들을 점검해 주세요.")
        
        self.results["overall_status"] = status
        
        print(f"\n📁 상세 결과는 validation_results.json에 저장됩니다.")
        
        # 결과 파일 저장
        with open("validation_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        return status in ["EXCELLENT", "GOOD"]

async def main():
    """메인 검증 실행"""
    validator = APIKeyValidator()
    
    try:
        success = await validator.run_validation()
        
        if success:
            print(f"\n✅ 검증 완료! AuroraQ v2.0가 성공적으로 설정되었습니다.")
            print(f"\n📋 다음 단계:")
            print(f"1. 실시간 뉴스 모니터링 시작")
            print(f"2. 거래 봇과 연동")
            print(f"3. Telegram 알림 확인")
        else:
            print(f"\n⚠️ 일부 문제가 발견되었습니다. validation_results.json을 확인하세요.")
    
    except KeyboardInterrupt:
        print(f"\n👋 검증이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 검증 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())