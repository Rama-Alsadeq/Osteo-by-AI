from chatbot.model import get_answer


def main():

    question = "What should I do to diagonis my knee if I have ostearthritis?"

    print(f" {question}\n")

    answer = get_answer(question)

    print(answer)


if __name__ == "__main__":
    main()
