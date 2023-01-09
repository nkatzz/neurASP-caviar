
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

	orientation(lbbcgt_4,137,881). coords(lbbcgt_4,9,62,881).  visible(lbbcgt_4,881). happensAt(walking(lbbcgt_4),881). orientation(lbbcgt_6,135,881). coords(lbbcgt_6,147,51,881).  visible(lbbcgt_6,881). happensAt(running(lbbcgt_6),881).
orientation(lbbcgt_4,137,882). coords(lbbcgt_4,9,61,882).  visible(lbbcgt_4,882). happensAt(walking(lbbcgt_4),882). orientation(lbbcgt_6,142,882). coords(lbbcgt_6,148,53,882).  visible(lbbcgt_6,882). happensAt(running(lbbcgt_6),882).
orientation(lbbcgt_4,132,883). coords(lbbcgt_4,9,61,883).  visible(lbbcgt_4,883). happensAt(walking(lbbcgt_4),883). orientation(lbbcgt_6,142,883). coords(lbbcgt_6,147,54,883).  visible(lbbcgt_6,883). happensAt(running(lbbcgt_6),883).
orientation(lbbcgt_4,132,884). coords(lbbcgt_4,10,60,884).  visible(lbbcgt_4,884). happensAt(walking(lbbcgt_4),884). orientation(lbbcgt_6,142,884). coords(lbbcgt_6,146,55,884).  visible(lbbcgt_6,884). happensAt(walking(lbbcgt_6),884).
orientation(lbbcgt_4,132,885). coords(lbbcgt_4,13,59,885).  visible(lbbcgt_4,885). happensAt(walking(lbbcgt_4),885). orientation(lbbcgt_6,142,885). coords(lbbcgt_6,145,56,885).  visible(lbbcgt_6,885). happensAt(walking(lbbcgt_6),885).
orientation(lbbcgt_4,132,886). coords(lbbcgt_4,13,57,886).  visible(lbbcgt_4,886). happensAt(walking(lbbcgt_4),886). orientation(lbbcgt_6,142,886). coords(lbbcgt_6,143,57,886).  visible(lbbcgt_6,886). happensAt(walking(lbbcgt_6),886).
orientation(lbbcgt_4,137,887). coords(lbbcgt_4,13,56,887).  visible(lbbcgt_4,887). happensAt(walking(lbbcgt_4),887). orientation(lbbcgt_6,142,887). coords(lbbcgt_6,142,58,887).  visible(lbbcgt_6,887). happensAt(walking(lbbcgt_6),887).
orientation(lbbcgt_4,137,888). coords(lbbcgt_4,13,55,888).  visible(lbbcgt_4,888). happensAt(walking(lbbcgt_4),888). orientation(lbbcgt_6,142,888). coords(lbbcgt_6,140,58,888).  visible(lbbcgt_6,888). happensAt(walking(lbbcgt_6),888).
orientation(lbbcgt_4,137,889). coords(lbbcgt_4,13,54,889).  visible(lbbcgt_4,889). happensAt(walking(lbbcgt_4),889). orientation(lbbcgt_6,142,889). coords(lbbcgt_6,138,58,889).  visible(lbbcgt_6,889). happensAt(walking(lbbcgt_6),889).
orientation(lbbcgt_4,137,890). coords(lbbcgt_4,13,54,890).  visible(lbbcgt_4,890). happensAt(walking(lbbcgt_4),890). orientation(lbbcgt_6,142,890). coords(lbbcgt_6,136,59,890).  visible(lbbcgt_6,890). happensAt(walking(lbbcgt_6),890).
orientation(lbbcgt_4,137,891). coords(lbbcgt_4,13,53,891).  happensAt(occluded(lbbcgt_4),891). happensAt(walking(lbbcgt_4),891). orientation(lbbcgt_6,142,891). coords(lbbcgt_6,135,60,891).  visible(lbbcgt_6,891). happensAt(walking(lbbcgt_6),891).
orientation(lbbcgt_4,137,892). coords(lbbcgt_4,13,53,892).  happensAt(occluded(lbbcgt_4),892). happensAt(walking(lbbcgt_4),892). orientation(lbbcgt_6,142,892). coords(lbbcgt_6,134,63,892).  visible(lbbcgt_6,892). happensAt(walking(lbbcgt_6),892).
orientation(lbbcgt_4,137,893). coords(lbbcgt_4,13,53,893).  happensAt(occluded(lbbcgt_4),893). happensAt(walking(lbbcgt_4),893). orientation(lbbcgt_6,142,893). coords(lbbcgt_6,134,64,893).  visible(lbbcgt_6,893). happensAt(walking(lbbcgt_6),893).
orientation(lbbcgt_4,137,894). coords(lbbcgt_4,13,53,894).  happensAt(occluded(lbbcgt_4),894). happensAt(walking(lbbcgt_4),894). orientation(lbbcgt_6,142,894). coords(lbbcgt_6,133,66,894).  visible(lbbcgt_6,894). happensAt(walking(lbbcgt_6),894).
orientation(lbbcgt_4,137,895). coords(lbbcgt_4,17,46,895).  visible(lbbcgt_4,895). happensAt(walking(lbbcgt_4),895). orientation(lbbcgt_6,142,895). coords(lbbcgt_6,133,67,895).  visible(lbbcgt_6,895). happensAt(walking(lbbcgt_6),895).
orientation(lbbcgt_4,140,896). coords(lbbcgt_4,17,46,896).  visible(lbbcgt_4,896). happensAt(walking(lbbcgt_4),896). orientation(lbbcgt_6,142,896). coords(lbbcgt_6,132,68,896).  visible(lbbcgt_6,896). happensAt(walking(lbbcgt_6),896).
orientation(lbbcgt_4,144,897). coords(lbbcgt_4,17,45,897).  visible(lbbcgt_4,897). happensAt(walking(lbbcgt_4),897). orientation(lbbcgt_6,142,897). coords(lbbcgt_6,132,68,897).  visible(lbbcgt_6,897). happensAt(walking(lbbcgt_6),897).
orientation(lbbcgt_4,146,898). coords(lbbcgt_4,17,45,898).  visible(lbbcgt_4,898). happensAt(walking(lbbcgt_4),898). orientation(lbbcgt_6,142,898). coords(lbbcgt_6,131,69,898).  visible(lbbcgt_6,898). happensAt(walking(lbbcgt_6),898).
orientation(lbbcgt_4,146,899). coords(lbbcgt_4,16,45,899).  visible(lbbcgt_4,899). happensAt(walking(lbbcgt_4),899). orientation(lbbcgt_6,142,899). coords(lbbcgt_6,129,70,899).  visible(lbbcgt_6,899). happensAt(walking(lbbcgt_6),899).
orientation(lbbcgt_4,146,900). coords(lbbcgt_4,16,45,900).  visible(lbbcgt_4,900). happensAt(walking(lbbcgt_4),900). orientation(lbbcgt_6,142,900). coords(lbbcgt_6,127,70,900).  visible(lbbcgt_6,900). happensAt(walking(lbbcgt_6),900).
orientation(lbbcgt_4,146,901). coords(lbbcgt_4,18,45,901).  visible(lbbcgt_4,901). happensAt(walking(lbbcgt_4),901). orientation(lbbcgt_6,142,901). coords(lbbcgt_6,125,70,901).  visible(lbbcgt_6,901). happensAt(walking(lbbcgt_6),901).
orientation(lbbcgt_4,146,902). coords(lbbcgt_4,18,45,902).  visible(lbbcgt_4,902). happensAt(walking(lbbcgt_4),902). orientation(lbbcgt_6,142,902). coords(lbbcgt_6,123,71,902).  visible(lbbcgt_6,902). happensAt(walking(lbbcgt_6),902).
orientation(lbbcgt_4,146,903). coords(lbbcgt_4,18,45,903).  visible(lbbcgt_4,903). happensAt(walking(lbbcgt_4),903). orientation(lbbcgt_6,142,903). coords(lbbcgt_6,122,71,903).  visible(lbbcgt_6,903). happensAt(walking(lbbcgt_6),903).
orientation(lbbcgt_4,146,904). coords(lbbcgt_4,18,45,904).  visible(lbbcgt_4,904). happensAt(walking(lbbcgt_4),904). orientation(lbbcgt_6,142,904). coords(lbbcgt_6,121,71,904).  visible(lbbcgt_6,904). happensAt(walking(lbbcgt_6),904).
orientation(lbbcgt_4,146,905). coords(lbbcgt_4,18,44,905).  happensAt(disappear(lbbcgt_4),905). happensAt(walking(lbbcgt_4),905). orientation(lbbcgt_6,142,905). coords(lbbcgt_6,121,72,905).  visible(lbbcgt_6,905). happensAt(walking(lbbcgt_6),905).
#show holdsAt/2.