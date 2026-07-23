# EXP-M5-8b — S5 top-k unembed tokens (what each steering vector points at)

- model EleutherAI/pythia-410m@9879c9b; top-15; representative layers [5, 10, 15]; logit lens = output space

Alignment is a HUMAN read of the raw tokens below (the authored-word flag is a convenience, not decisive).

## S5:sentiment
- **L5 logit**: nicely, compliment, and, complimentary, –, complementary, safe, pleasing, ya, complement, lik, excell, improved, sch, agree
- L5 jlens: celebrates, celebrating, nership, celebrate, marvel, celebrated, Gift, Celebr, gifts, cele, celebrations, beautifully, voyage, celebration, gift
- **L10 logit**: nicely, particip, thanks, dez, wonderful, gov, excell, ital, feat, ith, praise, pleasing, excellence, complimentary, pleased
- L10 jlens: nership, wonderful, marvel, showc, quo, gift, amazing, bonus, complimentary, CEPT, congrat, partnership, facilit, enrich, splendid
- **L15 logit**: versatile, versatility, appreciated, informative, appreciate, Excellent, goodness, pleased, excell, excellent, excellence, wonderful, splendid, marvel, pleasant
- L15 jlens: wonderful, marvel, beautifully, Excellent, excellent, Excellent, superb, versatile, wonderfully, splendid, appreciated, pleased, delightful, innovative, spacious
- convenience — authored pos_w in any top-20 logit: **['amazing', 'excellent', 'good', 'nice', 'wonderful']** (authored: ['good', 'great', 'wonderful', 'amazing', 'excellent', 'nice'])

## S5:formality
- **L5 logit**: agu, uge, sight, acknow, omo, refe, inform, LP, ounce, arity, uel, ér, anim, isse, =.
- L5 jlens: iduc, Cardinal, éné, Institutional, Trustees, fiduciary, Instit, [*, Protocol, igraph, Facility, Clause, Memorandum, Administrative, Service
- **L10 logit**: isan, ÐµÐ, ibase, above, cern, isse, iven, ül, uge, ivat, issen, inquire, ilder, ipel, ür
- L10 jlens: Pursuant, pursuant, ,—, Memorandum, entrusted, undertaking, éné, —, Protocol, accorded, Provincial, assistance, Municip, Petitioners, memorandum
- **L15 logit**: cknowled, :</, cknow, ––, answ, advertis, inquire, bsite, atheros, enqu, fprintf, :--, lapt, entrusted, acknow
- L15 jlens: Pursuant, pursuant, Securities, Administrative, Petitioners, Advisory, Superintendent, disclosures, inquiries, receipt, iduc, Institutional, advisory, Representative, Certificate
- convenience — authored pos_w in any top-20 logit: **NONE** (authored: ['therefore', 'furthermore', 'regarding', 'accordingly', 'moreover', 'hereby'])

## S5:politeness
- **L5 logit**: eed, please, ?’, referee, samples, ane, ��, gi, enable, authors, \]](, sample, reate, record, handling
- L5 jlens: chez, Rousseau, VERTIS, conjug, François, automorphism, ologous, favourable, estock, ž, �, ée, economists, generous, ORS
- **L10 logit**: resem, authors, resemblance, referee, vez, haus, comment, hosts, comment, iness, unusually, ünd, hani, yn, Representatives
- L10 jlens: ylene, donation, assistance, participants, recipients, hus, ünd, Petition, assisting, participant, Assistance, organis, petitioners, Petitioners, URL
- **L15 logit**: authors, orden, comment, representative, iek, please, kindly, oro, ylene, unusually, me, hov, rike, uru, erc
- L15 jlens: please, phot, donation, viewing, kindly, similar, assistance, assisting, hov, }?, HEL, recent, ?, request, permission
- convenience — authored pos_w in any top-20 logit: **['kindly', 'please', 'thank']** (authored: ['please', 'kindly', 'thank', 'appreciate', 'grateful', 'sorry'])

## S5:excitement
- **L5 logit**: !_, atz, "_, accord, atti, !', ause, !’, amaz, describ, estine, ocks, appl, EXT, pred
- L5 jlens: !, !!, !!!, !),, !), !!!!, !)., !', !, !!!!!!!!, ?!, ,, !,, IMP, !_
- **L10 logit**: "_, assurance, surprises, hope, doll, feat, freedom, iy, poder, amount, surprise, leverage, custody, physical, esis
- L10 jlens: !!, !!!, !, !!!!, �, !!, !),, !!!!!!!!, !, ！, 🙂, !', ?!, �, !,
- **L15 logit**: !!, !!!, !, ！, !!!!, !”, !', !_, !", )!, !’, !,, !!!!!!!!, hope, !(
- L15 jlens: !!, !!!, !, !!!!, !', !(, ！, !!!!!!!!, !_, !, !!, )!, !., !", !,
- convenience — authored pos_w in any top-20 logit: **['amazing', 'incredible']** (authored: ['thrilling', 'exciting', 'amazing', 'incredible', 'awesome', 'wow'])

wall 18.1 s; peak 1.97 GB.