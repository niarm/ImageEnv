import gym
import textEnv
from time import sleep

if __name__== "__main__":
    
    texts = ['US-Präsident Donald Trump verteidigt den Truppenabzug aus dem Norden Syriens und will "endlich aus diesen lächerlichen endlosen Kriegen" herauskommen. Die einstigen kurdischen Verbündeten sprechen von einem "Dolchstoß"',
             "Der Einmarsch der türkischen Armee in den Norden Syriens steht unmittelbar bevor - und damit eine erneute Eskalation der Kämpfe mit kurdischen Einheiten. Die US-Truppen sollen das Gebiet verlassen.",
             "Immer wieder hat der türkische Staatspräsident Recep Tayyip Erdogan in den vergangenen Monaten mit einer türkischen Offensive an der syrischen Grenze gedroht."]
    

    use_feature_extractor = textEnv.use_feature_extractor.USEFeatureExtractor()

    env = gym.make('TextSimilarityEnv-v0', feature_extractor=use_feature_extractor)
    env.set_texts(texts=texts)
    env.create_perception_fields(1)
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