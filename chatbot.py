import streamlit as st

import openai
import os
import pandas as pd
import ast

import numpy as np
from numpy import dot
from numpy.linalg import norm

from pprint import pprint

# conda install streamlit-chat -y
from streamlit_chat import message

openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key = openai_api_key)

def get_embedding(text):
    try :
        print("get_embedding :: api 호출을 시도합니다")
        response = client.embeddings.create(
            input=text,
            model='text-embedding-3-small'
        )
    except Exception as e:
        # 예외가 발생했을 때 처리할 코드
        print(e)
        return
    else:
        # 예외가 발생하지 않았을 때 실행할 코드
        print("api 응답을 받았습니다")
        return response.data[0].embedding

# folder_path와 file_name을 결합하여 file_path = './data/embedding.csv'
folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

# if: embedding.csv가 이미 존재한다면 데이터프레임 df로 로드한다.
if os.path.isfile(file_path):
    print("임베딩 파일이 존재합니다")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)


# 그렇지 않다면 text열과 embedding열이 존재하는 df를 신규 생성해야한다.
else:
    print("임베딩 파일이 존재하지 않습니다")
    # 서울 청년 정책 txt 파일명을 txt_files에 저장한다.
    print("데이터 파일을 읽어옵니다")
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    if txt_files is None or len(txt_files) == 0 :
        print("데이터 파일이 존재하지 않습니다")
    else :
        print("데이터 파일을 찾았습니다")
        data = []
        # txt_files로부터 청년 정책 데이터를 로드하여 df를 신규 생성한다.
        for file in txt_files:
            print("작업중인 파일 : ", file)
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                data.append(text)

        df = pd.DataFrame(data, columns=['text'])
        
        if df is not None :
            print("임배딩 생성중")
            # 데이터프레임의 text 열로부터 embedding열을 생성한다.
            df['embedding'] = df.apply(lambda row: get_embedding(
                row.text,
            ), axis=1)

            print("\n임배딩 생성 완료")

            # 추후 사용을 위해 df를 'embedding.csv' 파일로 저장한다.
            # 이렇게 저장되면 추후 실행에서는 df를 새로 만드는 과정을 생략한다.
            print("임베딩 파일을 생성합니다")
            try:
                # 예외가 발생할 수 있는 코드
                df.to_csv("./data/embedding.csv", index=False, encoding='utf-8')
                print("임베딩 파일을 생성합니다")
            except Exception as e:
                # 예외가 발생했을 때 처리할 코드
                print(e)
            else:
                print("파일 생성을 완료했습니다")
                # 예외가 발생하지 않았을 때 실행할 코드

# 주어진 질의로부터 유사한 문서 개를 반환하는 검색 시스템.
# 함수 return_answer_candidate내부에서 유사도 계산을 위해 cos_sim을 호출.
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    # query : 유저의 입력
    print("유저의 질문 : ", query)
    query_embedding = get_embedding(query)
    if query_embedding is not None :
        print("유저 질문의 임베딩 생성")
        df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
        # 유사도 측정 결과를 정렬하여 상위 3개를 뽑아낸다
        top_three_doc = df.sort_values("similarity", ascending=False).head(3)
        print("top_three_doc : ", top_three_doc)
        return top_three_doc

# 챗봇의 답변을 만들기 위해 사용될 프롬프트를 만드는 함수.
def create_prompt(df, query):
    # query : 유저의 입력
    result = return_answer_candidate(df, query)
    system_role = f"""You are an artificial intelligence language model named "정채기" that specializes in summarizing \
    and answering documents about Seoul's youth policy, developed by developer 홍길동.
    You need to take a given document and return a very detailed summary of the document in the query language.
    Here are the document: 
            doc 1 :{str(result.iloc[0]['text'])}
            doc 2 :{str(result.iloc[1]['text'])}
            doc 3 :{str(result.iloc[2]['text'])}
    You must return in Korean. Return a accurate answer based on the document.
    """
    user_content = f"""User question: "{str(query)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]
    return messages

# 위의 create_prompt 함수가 생성한 프롬프트로부터 챗봇의 답변을 만드는 함수.
def generate_response(messages):
    print("generate_response :: api 요청을 합니다")
    try :
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=500)
    except Exception as e :
        print(e)
        return
    else :
        print("generate_response :: 요청이 완료되었습니다")
        return result.choices[0].message.content


#st.image('images/ask_me_chatbot.png')

# 화면에 보여주기 위해 챗봇의 답변을 저장할 공간 할당
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자의 질문을 저장할 공간 할당
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자의 입력이 들어오면 user_input에 저장하고 Send 버튼을 클릭하면
# submitted의 값이 True로 변환.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('정책을 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

# submitted의 값이 True면 챗봇이 답변을 하기 시작
if submitted and user_input:
    # 프롬프트 생성
    try :
        prompt = create_prompt(df, user_input)
    except Exception as e :
        print(e)
    else :
        try :
            # 생성한 프롬프트를 기반으로 챗봇 답변을 생성
            chatbot_response = generate_response(prompt)
        except Exception as e :
            print(e)
        else :
            # 화면에 보여주기 위해 사용자의 질문과 챗봇의 답변을 각각 저장
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(chatbot_response)

# 챗봇의 답변이 있으면 사용자의 질문과 챗봇의 답변을 가장 최근의 순서로 화면에 출력
if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))