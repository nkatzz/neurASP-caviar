
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(wk2gt_2,142,884). coords(wk2gt_2,99,63,884).  visible(wk2gt_2,884). happensAt(inactive(wk2gt_2),884). orientation(wk2gt_5,156,884). coords(wk2gt_5,53,99,884).  visible(wk2gt_5,884). happensAt(walking(wk2gt_5),884).
orientation(wk2gt_2,142,885). coords(wk2gt_2,99,63,885).  visible(wk2gt_2,885). happensAt(inactive(wk2gt_2),885). orientation(wk2gt_5,156,885). coords(wk2gt_5,53,99,885).  visible(wk2gt_5,885). happensAt(walking(wk2gt_5),885).
orientation(wk2gt_2,142,886). coords(wk2gt_2,99,63,886).  visible(wk2gt_2,886). happensAt(inactive(wk2gt_2),886). orientation(wk2gt_5,156,886). coords(wk2gt_5,53,99,886).  visible(wk2gt_5,886). happensAt(walking(wk2gt_5),886).
orientation(wk2gt_2,142,887). coords(wk2gt_2,99,63,887).  visible(wk2gt_2,887). happensAt(inactive(wk2gt_2),887). orientation(wk2gt_5,156,887). coords(wk2gt_5,53,99,887).  visible(wk2gt_5,887). happensAt(walking(wk2gt_5),887).
orientation(wk2gt_2,142,888). coords(wk2gt_2,99,63,888).  visible(wk2gt_2,888). happensAt(inactive(wk2gt_2),888). orientation(wk2gt_5,156,888). coords(wk2gt_5,53,99,888).  visible(wk2gt_5,888). happensAt(walking(wk2gt_5),888).
orientation(wk2gt_2,142,889). coords(wk2gt_2,99,63,889).  visible(wk2gt_2,889). happensAt(inactive(wk2gt_2),889). orientation(wk2gt_5,156,889). coords(wk2gt_5,52,97,889).  visible(wk2gt_5,889). happensAt(walking(wk2gt_5),889).
orientation(wk2gt_2,142,890). coords(wk2gt_2,99,63,890).  visible(wk2gt_2,890). happensAt(inactive(wk2gt_2),890). orientation(wk2gt_5,156,890). coords(wk2gt_5,52,97,890).  visible(wk2gt_5,890). happensAt(walking(wk2gt_5),890).
orientation(wk2gt_2,142,891). coords(wk2gt_2,99,63,891).  visible(wk2gt_2,891). happensAt(inactive(wk2gt_2),891). orientation(wk2gt_5,156,891). coords(wk2gt_5,55,98,891).  visible(wk2gt_5,891). happensAt(walking(wk2gt_5),891).
orientation(wk2gt_2,142,892). coords(wk2gt_2,99,63,892).  visible(wk2gt_2,892). happensAt(inactive(wk2gt_2),892). orientation(wk2gt_5,156,892). coords(wk2gt_5,56,98,892).  visible(wk2gt_5,892). happensAt(walking(wk2gt_5),892).
orientation(wk2gt_2,142,893). coords(wk2gt_2,99,63,893).  visible(wk2gt_2,893). happensAt(inactive(wk2gt_2),893). orientation(wk2gt_5,156,893). coords(wk2gt_5,56,97,893).  visible(wk2gt_5,893). happensAt(walking(wk2gt_5),893).
orientation(wk2gt_2,142,894). coords(wk2gt_2,99,63,894).  visible(wk2gt_2,894). happensAt(inactive(wk2gt_2),894). orientation(wk2gt_5,156,894). coords(wk2gt_5,57,97,894).  visible(wk2gt_5,894). happensAt(walking(wk2gt_5),894).
orientation(wk2gt_2,142,895). coords(wk2gt_2,99,63,895).  visible(wk2gt_2,895). happensAt(inactive(wk2gt_2),895). orientation(wk2gt_5,156,895). coords(wk2gt_5,57,97,895).  visible(wk2gt_5,895). happensAt(walking(wk2gt_5),895).
orientation(wk2gt_2,142,896). coords(wk2gt_2,99,63,896).  visible(wk2gt_2,896). happensAt(inactive(wk2gt_2),896). orientation(wk2gt_5,156,896). coords(wk2gt_5,57,97,896).  visible(wk2gt_5,896). happensAt(walking(wk2gt_5),896).
orientation(wk2gt_2,142,897). coords(wk2gt_2,99,63,897).  visible(wk2gt_2,897). happensAt(inactive(wk2gt_2),897). orientation(wk2gt_5,156,897). coords(wk2gt_5,57,97,897).  visible(wk2gt_5,897). happensAt(walking(wk2gt_5),897).
orientation(wk2gt_2,142,898). coords(wk2gt_2,99,63,898).  visible(wk2gt_2,898). happensAt(inactive(wk2gt_2),898). orientation(wk2gt_5,156,898). coords(wk2gt_5,57,97,898).  visible(wk2gt_5,898). happensAt(walking(wk2gt_5),898).
orientation(wk2gt_2,142,899). coords(wk2gt_2,99,63,899).  visible(wk2gt_2,899). happensAt(inactive(wk2gt_2),899). orientation(wk2gt_5,156,899). coords(wk2gt_5,57,97,899).  visible(wk2gt_5,899). happensAt(walking(wk2gt_5),899).
orientation(wk2gt_2,142,900). coords(wk2gt_2,99,63,900).  visible(wk2gt_2,900). happensAt(inactive(wk2gt_2),900). orientation(wk2gt_5,156,900). coords(wk2gt_5,58,97,900).  visible(wk2gt_5,900). happensAt(walking(wk2gt_5),900).
orientation(wk2gt_2,142,901). coords(wk2gt_2,99,63,901).  visible(wk2gt_2,901). happensAt(inactive(wk2gt_2),901). orientation(wk2gt_5,156,901). coords(wk2gt_5,59,97,901).  visible(wk2gt_5,901). happensAt(walking(wk2gt_5),901).
orientation(wk2gt_2,142,902). coords(wk2gt_2,99,63,902).  visible(wk2gt_2,902). happensAt(inactive(wk2gt_2),902). orientation(wk2gt_5,156,902). coords(wk2gt_5,59,97,902).  visible(wk2gt_5,902). happensAt(walking(wk2gt_5),902).
orientation(wk2gt_2,142,903). coords(wk2gt_2,99,63,903).  visible(wk2gt_2,903). happensAt(inactive(wk2gt_2),903). orientation(wk2gt_5,156,903). coords(wk2gt_5,60,96,903).  visible(wk2gt_5,903). happensAt(walking(wk2gt_5),903).
orientation(wk2gt_2,142,904). coords(wk2gt_2,99,63,904).  visible(wk2gt_2,904). happensAt(inactive(wk2gt_2),904). orientation(wk2gt_5,156,904). coords(wk2gt_5,61,96,904).  visible(wk2gt_5,904). happensAt(walking(wk2gt_5),904).
orientation(wk2gt_2,142,905). coords(wk2gt_2,99,63,905).  visible(wk2gt_2,905). happensAt(inactive(wk2gt_2),905). orientation(wk2gt_5,156,905). coords(wk2gt_5,61,96,905).  visible(wk2gt_5,905). happensAt(walking(wk2gt_5),905).
orientation(wk2gt_2,142,906). coords(wk2gt_2,99,63,906).  visible(wk2gt_2,906). happensAt(inactive(wk2gt_2),906). orientation(wk2gt_5,156,906). coords(wk2gt_5,61,96,906).  visible(wk2gt_5,906). happensAt(walking(wk2gt_5),906).
orientation(wk2gt_2,142,907). coords(wk2gt_2,99,63,907).  visible(wk2gt_2,907). happensAt(inactive(wk2gt_2),907). orientation(wk2gt_5,156,907). coords(wk2gt_5,61,96,907).  visible(wk2gt_5,907). happensAt(walking(wk2gt_5),907).
orientation(wk2gt_2,142,908). coords(wk2gt_2,99,63,908).  visible(wk2gt_2,908). happensAt(inactive(wk2gt_2),908). orientation(wk2gt_5,156,908). coords(wk2gt_5,61,96,908).  visible(wk2gt_5,908). happensAt(walking(wk2gt_5),908).
#show holdsAt/2.