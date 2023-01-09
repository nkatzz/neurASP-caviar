
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

	orientation(lbbcgt_4,137,881). coords(lbbcgt_4,9,62,881).  visible(lbbcgt_4,881). happensAt(walking(lbbcgt_4),881). orientation(lbbcgt_5,64,881). coords(lbbcgt_5,249,281,881).  visible(lbbcgt_5,881). happensAt(walking(lbbcgt_5),881).
orientation(lbbcgt_4,137,882). coords(lbbcgt_4,9,61,882).  visible(lbbcgt_4,882). happensAt(walking(lbbcgt_4),882). orientation(lbbcgt_5,64,882). coords(lbbcgt_5,249,281,882).  visible(lbbcgt_5,882). happensAt(walking(lbbcgt_5),882).
orientation(lbbcgt_4,132,883). coords(lbbcgt_4,9,61,883).  visible(lbbcgt_4,883). happensAt(walking(lbbcgt_4),883). orientation(lbbcgt_5,64,883). coords(lbbcgt_5,248,281,883).  visible(lbbcgt_5,883). happensAt(walking(lbbcgt_5),883).
orientation(lbbcgt_4,132,884). coords(lbbcgt_4,10,60,884).  visible(lbbcgt_4,884). happensAt(walking(lbbcgt_4),884). orientation(lbbcgt_5,20,884). coords(lbbcgt_5,243,281,884).  visible(lbbcgt_5,884). happensAt(walking(lbbcgt_5),884).
orientation(lbbcgt_4,132,885). coords(lbbcgt_4,13,59,885).  visible(lbbcgt_4,885). happensAt(walking(lbbcgt_4),885). orientation(lbbcgt_5,20,885). coords(lbbcgt_5,241,280,885).  visible(lbbcgt_5,885). happensAt(walking(lbbcgt_5),885).
orientation(lbbcgt_4,132,886). coords(lbbcgt_4,13,57,886).  visible(lbbcgt_4,886). happensAt(walking(lbbcgt_4),886). orientation(lbbcgt_5,20,886). coords(lbbcgt_5,239,278,886).  visible(lbbcgt_5,886). happensAt(walking(lbbcgt_5),886).
orientation(lbbcgt_4,137,887). coords(lbbcgt_4,13,56,887).  visible(lbbcgt_4,887). happensAt(walking(lbbcgt_4),887). orientation(lbbcgt_5,20,887). coords(lbbcgt_5,238,274,887).  visible(lbbcgt_5,887). happensAt(walking(lbbcgt_5),887).
orientation(lbbcgt_4,137,888). coords(lbbcgt_4,13,55,888).  visible(lbbcgt_4,888). happensAt(walking(lbbcgt_4),888). orientation(lbbcgt_5,20,888). coords(lbbcgt_5,237,271,888).  visible(lbbcgt_5,888). happensAt(walking(lbbcgt_5),888).
orientation(lbbcgt_4,137,889). coords(lbbcgt_4,13,54,889).  visible(lbbcgt_4,889). happensAt(walking(lbbcgt_4),889). orientation(lbbcgt_5,20,889). coords(lbbcgt_5,236,268,889).  visible(lbbcgt_5,889). happensAt(walking(lbbcgt_5),889).
orientation(lbbcgt_4,137,890). coords(lbbcgt_4,13,54,890).  visible(lbbcgt_4,890). happensAt(walking(lbbcgt_4),890). orientation(lbbcgt_5,20,890). coords(lbbcgt_5,236,267,890).  visible(lbbcgt_5,890). happensAt(walking(lbbcgt_5),890).
orientation(lbbcgt_4,137,891). coords(lbbcgt_4,13,53,891).  happensAt(occluded(lbbcgt_4),891). happensAt(walking(lbbcgt_4),891). orientation(lbbcgt_5,21,891). coords(lbbcgt_5,236,266,891).  visible(lbbcgt_5,891). happensAt(walking(lbbcgt_5),891).
orientation(lbbcgt_4,137,892). coords(lbbcgt_4,13,53,892).  happensAt(occluded(lbbcgt_4),892). happensAt(walking(lbbcgt_4),892). orientation(lbbcgt_5,21,892). coords(lbbcgt_5,236,266,892).  visible(lbbcgt_5,892). happensAt(walking(lbbcgt_5),892).
orientation(lbbcgt_4,137,893). coords(lbbcgt_4,13,53,893).  happensAt(occluded(lbbcgt_4),893). happensAt(walking(lbbcgt_4),893). orientation(lbbcgt_5,21,893). coords(lbbcgt_5,235,265,893).  visible(lbbcgt_5,893). happensAt(walking(lbbcgt_5),893).
orientation(lbbcgt_4,137,894). coords(lbbcgt_4,13,53,894).  happensAt(occluded(lbbcgt_4),894). happensAt(walking(lbbcgt_4),894). orientation(lbbcgt_5,21,894). coords(lbbcgt_5,234,264,894).  visible(lbbcgt_5,894). happensAt(walking(lbbcgt_5),894).
orientation(lbbcgt_4,137,895). coords(lbbcgt_4,17,46,895).  visible(lbbcgt_4,895). happensAt(walking(lbbcgt_4),895). orientation(lbbcgt_5,21,895). coords(lbbcgt_5,234,263,895).  visible(lbbcgt_5,895). happensAt(walking(lbbcgt_5),895).
orientation(lbbcgt_4,140,896). coords(lbbcgt_4,17,46,896).  visible(lbbcgt_4,896). happensAt(walking(lbbcgt_4),896). orientation(lbbcgt_5,21,896). coords(lbbcgt_5,231,261,896).  visible(lbbcgt_5,896). happensAt(walking(lbbcgt_5),896).
orientation(lbbcgt_4,144,897). coords(lbbcgt_4,17,45,897).  visible(lbbcgt_4,897). happensAt(walking(lbbcgt_4),897). orientation(lbbcgt_5,21,897). coords(lbbcgt_5,231,259,897).  visible(lbbcgt_5,897). happensAt(walking(lbbcgt_5),897).
orientation(lbbcgt_4,146,898). coords(lbbcgt_4,17,45,898).  visible(lbbcgt_4,898). happensAt(walking(lbbcgt_4),898). orientation(lbbcgt_5,9,898). coords(lbbcgt_5,230,257,898).  visible(lbbcgt_5,898). happensAt(walking(lbbcgt_5),898).
orientation(lbbcgt_4,146,899). coords(lbbcgt_4,16,45,899).  visible(lbbcgt_4,899). happensAt(walking(lbbcgt_4),899). orientation(lbbcgt_5,9,899). coords(lbbcgt_5,229,256,899).  visible(lbbcgt_5,899). happensAt(walking(lbbcgt_5),899).
orientation(lbbcgt_4,146,900). coords(lbbcgt_4,16,45,900).  visible(lbbcgt_4,900). happensAt(walking(lbbcgt_4),900). orientation(lbbcgt_5,9,900). coords(lbbcgt_5,229,255,900).  visible(lbbcgt_5,900). happensAt(walking(lbbcgt_5),900).
orientation(lbbcgt_4,146,901). coords(lbbcgt_4,18,45,901).  visible(lbbcgt_4,901). happensAt(walking(lbbcgt_4),901). orientation(lbbcgt_5,9,901). coords(lbbcgt_5,229,253,901).  visible(lbbcgt_5,901). happensAt(walking(lbbcgt_5),901).
orientation(lbbcgt_4,146,902). coords(lbbcgt_4,18,45,902).  visible(lbbcgt_4,902). happensAt(walking(lbbcgt_4),902). orientation(lbbcgt_5,9,902). coords(lbbcgt_5,228,251,902).  visible(lbbcgt_5,902). happensAt(walking(lbbcgt_5),902).
orientation(lbbcgt_4,146,903). coords(lbbcgt_4,18,45,903).  visible(lbbcgt_4,903). happensAt(walking(lbbcgt_4),903). orientation(lbbcgt_5,9,903). coords(lbbcgt_5,228,247,903).  visible(lbbcgt_5,903). happensAt(walking(lbbcgt_5),903).
orientation(lbbcgt_4,146,904). coords(lbbcgt_4,18,45,904).  visible(lbbcgt_4,904). happensAt(walking(lbbcgt_4),904). orientation(lbbcgt_5,9,904). coords(lbbcgt_5,227,244,904).  visible(lbbcgt_5,904). happensAt(walking(lbbcgt_5),904).
orientation(lbbcgt_4,146,905). coords(lbbcgt_4,18,44,905).  happensAt(disappear(lbbcgt_4),905). happensAt(walking(lbbcgt_4),905). orientation(lbbcgt_5,9,905). coords(lbbcgt_5,227,242,905).  visible(lbbcgt_5,905). happensAt(walking(lbbcgt_5),905).
#show holdsAt/2.