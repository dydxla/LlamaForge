from typing import List, Dict
from .templates import \
    (get_boolq_templates,
     get_squad_templates)

class PromptManager:
    def __init__(self):
        # prompt templates
        self.templates = {
            'boolq': get_boolq_templates(),
            'squad': get_squad_templates(),
            }

    def list_benchmarks(self,) -> List[str]:
        """
        가능한 벤치마크들을 리스트로 가져오는 메서드.

        Returns:
            List[str]: 입력가능한 벤치마크들
        """
        return list(self.templates.keys())
    
    def add_template(self, benchmark: str, prompt_type: str, template: str):
        """
        템플릿 목록에 원하는 템플릿을 추가하는 메서드.

        Args:
            benchmark (str): 추가하고자 하는 템플릿의 벤치마크 이름 (예: 'boolq')
            prompt_type (str): 프롬프트 유형 (예: 'basic', 'simple', 'detailed', 'fewshot-cot')
            
        Returns:
            Dict

        """
        if benchmark not in self.templates:
            self.templates[benchmark] = {}
        if prompt_type not in self.templates[benchmark]:
            self.templates[benchmark][prompt_type] = {}
        self.templates[benchmark][prompt_type] = template

    def get_templates(self, benchmark: str) -> Dict:
        """
        해당 벤치마크의 사용가능한 템플릿들을 가져오는 메서드.
        
        Args:
            benchmark (str): 벤치마크 이름 (예: 'boolq')

        Returns:
            Dict: 요청된 벤치마크의 모든 템플릿들
        """
        return self.templates[benchmark]

    def get_prompt(self, benchmark: str, prompt_type: str, **kwargs) -> str:
        """
        프롬프트를 가져오는 메서드.

        Args:
            benchmark (str): 벤치마크 이름 (예: 'boolq')
            prompt_type (str): 프롬프트 유형 (예: 'basic', 'simple', 'detailed', 'fewshot-cot')
            **kwargs: 템플릿에 삽입할 변수들 (question, passage 등)

        Returns:
            str: 요청된 프롬프트
        """
        try:
            # 선택된 벤치마크와 유형의 템플릿 가져오기
            template = self.templates[benchmark][prompt_type]
            # 템플릿에 변수 삽입
            return template.format(**kwargs)
        except KeyError:
            raise ValueError(f"Invalid benchmark ({benchmark}) or prompt type ({prompt_type}).")