import gym
import textEnv
from time import sleep

if __name__== "__main__":
    env = gym.make('TextSimilarityEnv-v0')
    
    texts = ["Dieser ist der erste kleine Text zum testen",
             "Das hier ist ein weiterer kurzer Text.",
             "Auch dieser Text erscheint mir recht kurz, zum testen jedoch geeignet.",
             "Jaja, deine Mudder... immer das Gleiche"]
    
    env.set_texts(texts=texts)
    env.set_n_perception_fields(1)
    env.next_text_pair()

    step_count = 0
    while True:
        env.step()
        #print("step #", count)

        step_count += 1
        env.render()

        if step_count > 1000:
            print(f"NEXT PAIR: {env.current_texts}")
            step_count = 0
            pairs_left = env.next_text_pair()
            if pairs_left == False:
                print(f"ENV DONE")
                break