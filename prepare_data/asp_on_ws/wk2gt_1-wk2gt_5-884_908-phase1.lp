
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

	orientation(wk2gt_1,127,884). coords(wk2gt_1,115,53,884).  visible(wk2gt_1,884). happensAt(inactive(wk2gt_1),884). orientation(wk2gt_5,156,884). coords(wk2gt_5,53,99,884).  visible(wk2gt_5,884). happensAt(walking(wk2gt_5),884).
orientation(wk2gt_1,127,885). coords(wk2gt_1,115,53,885).  visible(wk2gt_1,885). happensAt(inactive(wk2gt_1),885). orientation(wk2gt_5,156,885). coords(wk2gt_5,53,99,885).  visible(wk2gt_5,885). happensAt(walking(wk2gt_5),885).
orientation(wk2gt_1,127,886). coords(wk2gt_1,115,53,886).  visible(wk2gt_1,886). happensAt(inactive(wk2gt_1),886). orientation(wk2gt_5,156,886). coords(wk2gt_5,53,99,886).  visible(wk2gt_5,886). happensAt(walking(wk2gt_5),886).
orientation(wk2gt_1,127,887). coords(wk2gt_1,115,53,887).  visible(wk2gt_1,887). happensAt(inactive(wk2gt_1),887). orientation(wk2gt_5,156,887). coords(wk2gt_5,53,99,887).  visible(wk2gt_5,887). happensAt(walking(wk2gt_5),887).
orientation(wk2gt_1,127,888). coords(wk2gt_1,115,53,888).  visible(wk2gt_1,888). happensAt(inactive(wk2gt_1),888). orientation(wk2gt_5,156,888). coords(wk2gt_5,53,99,888).  visible(wk2gt_5,888). happensAt(walking(wk2gt_5),888).
orientation(wk2gt_1,127,889). coords(wk2gt_1,115,53,889).  visible(wk2gt_1,889). happensAt(inactive(wk2gt_1),889). orientation(wk2gt_5,156,889). coords(wk2gt_5,52,97,889).  visible(wk2gt_5,889). happensAt(walking(wk2gt_5),889).
orientation(wk2gt_1,127,890). coords(wk2gt_1,115,53,890).  visible(wk2gt_1,890). happensAt(inactive(wk2gt_1),890). orientation(wk2gt_5,156,890). coords(wk2gt_5,52,97,890).  visible(wk2gt_5,890). happensAt(walking(wk2gt_5),890).
orientation(wk2gt_1,127,891). coords(wk2gt_1,115,53,891).  visible(wk2gt_1,891). happensAt(inactive(wk2gt_1),891). orientation(wk2gt_5,156,891). coords(wk2gt_5,55,98,891).  visible(wk2gt_5,891). happensAt(walking(wk2gt_5),891).
orientation(wk2gt_1,127,892). coords(wk2gt_1,115,53,892).  visible(wk2gt_1,892). happensAt(inactive(wk2gt_1),892). orientation(wk2gt_5,156,892). coords(wk2gt_5,56,98,892).  visible(wk2gt_5,892). happensAt(walking(wk2gt_5),892).
orientation(wk2gt_1,127,893). coords(wk2gt_1,115,53,893).  visible(wk2gt_1,893). happensAt(inactive(wk2gt_1),893). orientation(wk2gt_5,156,893). coords(wk2gt_5,56,97,893).  visible(wk2gt_5,893). happensAt(walking(wk2gt_5),893).
orientation(wk2gt_1,127,894). coords(wk2gt_1,115,53,894).  visible(wk2gt_1,894). happensAt(inactive(wk2gt_1),894). orientation(wk2gt_5,156,894). coords(wk2gt_5,57,97,894).  visible(wk2gt_5,894). happensAt(walking(wk2gt_5),894).
orientation(wk2gt_1,127,895). coords(wk2gt_1,115,53,895).  visible(wk2gt_1,895). happensAt(inactive(wk2gt_1),895). orientation(wk2gt_5,156,895). coords(wk2gt_5,57,97,895).  visible(wk2gt_5,895). happensAt(walking(wk2gt_5),895).
orientation(wk2gt_1,127,896). coords(wk2gt_1,115,53,896).  visible(wk2gt_1,896). happensAt(inactive(wk2gt_1),896). orientation(wk2gt_5,156,896). coords(wk2gt_5,57,97,896).  visible(wk2gt_5,896). happensAt(walking(wk2gt_5),896).
orientation(wk2gt_1,127,897). coords(wk2gt_1,115,53,897).  visible(wk2gt_1,897). happensAt(inactive(wk2gt_1),897). orientation(wk2gt_5,156,897). coords(wk2gt_5,57,97,897).  visible(wk2gt_5,897). happensAt(walking(wk2gt_5),897).
orientation(wk2gt_1,127,898). coords(wk2gt_1,115,53,898).  visible(wk2gt_1,898). happensAt(inactive(wk2gt_1),898). orientation(wk2gt_5,156,898). coords(wk2gt_5,57,97,898).  visible(wk2gt_5,898). happensAt(walking(wk2gt_5),898).
orientation(wk2gt_1,127,899). coords(wk2gt_1,115,53,899).  visible(wk2gt_1,899). happensAt(inactive(wk2gt_1),899). orientation(wk2gt_5,156,899). coords(wk2gt_5,57,97,899).  visible(wk2gt_5,899). happensAt(walking(wk2gt_5),899).
orientation(wk2gt_1,127,900). coords(wk2gt_1,115,53,900).  visible(wk2gt_1,900). happensAt(inactive(wk2gt_1),900). orientation(wk2gt_5,156,900). coords(wk2gt_5,58,97,900).  visible(wk2gt_5,900). happensAt(walking(wk2gt_5),900).
orientation(wk2gt_1,127,901). coords(wk2gt_1,115,53,901).  visible(wk2gt_1,901). happensAt(inactive(wk2gt_1),901). orientation(wk2gt_5,156,901). coords(wk2gt_5,59,97,901).  visible(wk2gt_5,901). happensAt(walking(wk2gt_5),901).
orientation(wk2gt_1,127,902). coords(wk2gt_1,115,53,902).  visible(wk2gt_1,902). happensAt(inactive(wk2gt_1),902). orientation(wk2gt_5,156,902). coords(wk2gt_5,59,97,902).  visible(wk2gt_5,902). happensAt(walking(wk2gt_5),902).
orientation(wk2gt_1,127,903). coords(wk2gt_1,115,53,903).  visible(wk2gt_1,903). happensAt(inactive(wk2gt_1),903). orientation(wk2gt_5,156,903). coords(wk2gt_5,60,96,903).  visible(wk2gt_5,903). happensAt(walking(wk2gt_5),903).
orientation(wk2gt_1,127,904). coords(wk2gt_1,115,53,904).  visible(wk2gt_1,904). happensAt(inactive(wk2gt_1),904). orientation(wk2gt_5,156,904). coords(wk2gt_5,61,96,904).  visible(wk2gt_5,904). happensAt(walking(wk2gt_5),904).
orientation(wk2gt_1,127,905). coords(wk2gt_1,115,53,905).  visible(wk2gt_1,905). happensAt(inactive(wk2gt_1),905). orientation(wk2gt_5,156,905). coords(wk2gt_5,61,96,905).  visible(wk2gt_5,905). happensAt(walking(wk2gt_5),905).
orientation(wk2gt_1,127,906). coords(wk2gt_1,115,53,906).  visible(wk2gt_1,906). happensAt(inactive(wk2gt_1),906). orientation(wk2gt_5,156,906). coords(wk2gt_5,61,96,906).  visible(wk2gt_5,906). happensAt(walking(wk2gt_5),906).
orientation(wk2gt_1,127,907). coords(wk2gt_1,115,53,907).  visible(wk2gt_1,907). happensAt(inactive(wk2gt_1),907). orientation(wk2gt_5,156,907). coords(wk2gt_5,61,96,907).  visible(wk2gt_5,907). happensAt(walking(wk2gt_5),907).
orientation(wk2gt_1,124,908). coords(wk2gt_1,115,52,908).  visible(wk2gt_1,908). happensAt(inactive(wk2gt_1),908). orientation(wk2gt_5,156,908). coords(wk2gt_5,61,96,908).  visible(wk2gt_5,908). happensAt(walking(wk2gt_5),908).
#show holdsAt/2.