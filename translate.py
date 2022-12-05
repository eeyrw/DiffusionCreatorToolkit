from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt",cache_dir='.')
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt",cache_dir='.')
# translate Hindi to English
tokenizer.src_lang = "zh_CN"
encoded_hi = tokenizer('本周四，小米13系列将与U!I!> 14同台发布，手机已经开启了预热环节，MIUI 14这边也提前带来了尝鲜计划。据小米社区官方介绍，本次尝鲜计划活动时间为11月28日13:14-12月28日23:59，用户可在小米社区报名参与MIUI 14尝鲜计划，活动仅限选择一台小米设备报名（包括但不限于手机、平板）。', return_tensors="pt").to('cuda')
model.to('cuda')
generated_tokens = model.generate(**encoded_hi)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
