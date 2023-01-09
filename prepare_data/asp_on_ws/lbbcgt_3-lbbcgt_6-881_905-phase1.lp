
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

	orientation(lbbcgt_3,163,881). coords(lbbcgt_3,42,139,881).  visible(lbbcgt_3,881). happensAt(walking(lbbcgt_3),881). orientation(lbbcgt_6,135,881). coords(lbbcgt_6,147,51,881).  visible(lbbcgt_6,881). happensAt(running(lbbcgt_6),881).
orientation(lbbcgt_3,163,882). coords(lbbcgt_3,42,139,882).  visible(lbbcgt_3,882). happensAt(walking(lbbcgt_3),882). orientation(lbbcgt_6,142,882). coords(lbbcgt_6,148,53,882).  visible(lbbcgt_6,882). happensAt(running(lbbcgt_6),882).
orientation(lbbcgt_3,163,883). coords(lbbcgt_3,41,139,883).  visible(lbbcgt_3,883). happensAt(walking(lbbcgt_3),883). orientation(lbbcgt_6,142,883). coords(lbbcgt_6,147,54,883).  visible(lbbcgt_6,883). happensAt(running(lbbcgt_6),883).
orientation(lbbcgt_3,163,884). coords(lbbcgt_3,41,140,884).  visible(lbbcgt_3,884). happensAt(walking(lbbcgt_3),884). orientation(lbbcgt_6,142,884). coords(lbbcgt_6,146,55,884).  visible(lbbcgt_6,884). happensAt(walking(lbbcgt_6),884).
orientation(lbbcgt_3,163,885). coords(lbbcgt_3,41,140,885).  visible(lbbcgt_3,885). happensAt(walking(lbbcgt_3),885). orientation(lbbcgt_6,142,885). coords(lbbcgt_6,145,56,885).  visible(lbbcgt_6,885). happensAt(walking(lbbcgt_6),885).
orientation(lbbcgt_3,163,886). coords(lbbcgt_3,39,142,886).  visible(lbbcgt_3,886). happensAt(walking(lbbcgt_3),886). orientation(lbbcgt_6,142,886). coords(lbbcgt_6,143,57,886).  visible(lbbcgt_6,886). happensAt(walking(lbbcgt_6),886).
orientation(lbbcgt_3,163,887). coords(lbbcgt_3,39,142,887).  visible(lbbcgt_3,887). happensAt(walking(lbbcgt_3),887). orientation(lbbcgt_6,142,887). coords(lbbcgt_6,142,58,887).  visible(lbbcgt_6,887). happensAt(walking(lbbcgt_6),887).
orientation(lbbcgt_3,163,888). coords(lbbcgt_3,38,144,888).  visible(lbbcgt_3,888). happensAt(walking(lbbcgt_3),888). orientation(lbbcgt_6,142,888). coords(lbbcgt_6,140,58,888).  visible(lbbcgt_6,888). happensAt(walking(lbbcgt_6),888).
orientation(lbbcgt_3,163,889). coords(lbbcgt_3,39,144,889).  visible(lbbcgt_3,889). happensAt(walking(lbbcgt_3),889). orientation(lbbcgt_6,142,889). coords(lbbcgt_6,138,58,889).  visible(lbbcgt_6,889). happensAt(walking(lbbcgt_6),889).
orientation(lbbcgt_3,163,890). coords(lbbcgt_3,38,144,890).  visible(lbbcgt_3,890). happensAt(walking(lbbcgt_3),890). orientation(lbbcgt_6,142,890). coords(lbbcgt_6,136,59,890).  visible(lbbcgt_6,890). happensAt(walking(lbbcgt_6),890).
orientation(lbbcgt_3,163,891). coords(lbbcgt_3,39,146,891).  visible(lbbcgt_3,891). happensAt(walking(lbbcgt_3),891). orientation(lbbcgt_6,142,891). coords(lbbcgt_6,135,60,891).  visible(lbbcgt_6,891). happensAt(walking(lbbcgt_6),891).
orientation(lbbcgt_3,163,892). coords(lbbcgt_3,39,148,892).  visible(lbbcgt_3,892). happensAt(walking(lbbcgt_3),892). orientation(lbbcgt_6,142,892). coords(lbbcgt_6,134,63,892).  visible(lbbcgt_6,892). happensAt(walking(lbbcgt_6),892).
orientation(lbbcgt_3,163,893). coords(lbbcgt_3,39,148,893).  visible(lbbcgt_3,893). happensAt(walking(lbbcgt_3),893). orientation(lbbcgt_6,142,893). coords(lbbcgt_6,134,64,893).  visible(lbbcgt_6,893). happensAt(walking(lbbcgt_6),893).
orientation(lbbcgt_3,163,894). coords(lbbcgt_3,39,149,894).  visible(lbbcgt_3,894). happensAt(walking(lbbcgt_3),894). orientation(lbbcgt_6,142,894). coords(lbbcgt_6,133,66,894).  visible(lbbcgt_6,894). happensAt(walking(lbbcgt_6),894).
orientation(lbbcgt_3,163,895). coords(lbbcgt_3,39,149,895).  visible(lbbcgt_3,895). happensAt(walking(lbbcgt_3),895). orientation(lbbcgt_6,142,895). coords(lbbcgt_6,133,67,895).  visible(lbbcgt_6,895). happensAt(walking(lbbcgt_6),895).
orientation(lbbcgt_3,163,896). coords(lbbcgt_3,39,149,896).  visible(lbbcgt_3,896). happensAt(walking(lbbcgt_3),896). orientation(lbbcgt_6,142,896). coords(lbbcgt_6,132,68,896).  visible(lbbcgt_6,896). happensAt(walking(lbbcgt_6),896).
orientation(lbbcgt_3,163,897). coords(lbbcgt_3,39,150,897).  visible(lbbcgt_3,897). happensAt(walking(lbbcgt_3),897). orientation(lbbcgt_6,142,897). coords(lbbcgt_6,132,68,897).  visible(lbbcgt_6,897). happensAt(walking(lbbcgt_6),897).
orientation(lbbcgt_3,163,898). coords(lbbcgt_3,39,151,898).  visible(lbbcgt_3,898). happensAt(walking(lbbcgt_3),898). orientation(lbbcgt_6,142,898). coords(lbbcgt_6,131,69,898).  visible(lbbcgt_6,898). happensAt(walking(lbbcgt_6),898).
orientation(lbbcgt_3,163,899). coords(lbbcgt_3,39,151,899).  visible(lbbcgt_3,899). happensAt(walking(lbbcgt_3),899). orientation(lbbcgt_6,142,899). coords(lbbcgt_6,129,70,899).  visible(lbbcgt_6,899). happensAt(walking(lbbcgt_6),899).
orientation(lbbcgt_3,163,900). coords(lbbcgt_3,39,151,900).  visible(lbbcgt_3,900). happensAt(walking(lbbcgt_3),900). orientation(lbbcgt_6,142,900). coords(lbbcgt_6,127,70,900).  visible(lbbcgt_6,900). happensAt(walking(lbbcgt_6),900).
orientation(lbbcgt_3,163,901). coords(lbbcgt_3,39,152,901).  visible(lbbcgt_3,901). happensAt(walking(lbbcgt_3),901). orientation(lbbcgt_6,142,901). coords(lbbcgt_6,125,70,901).  visible(lbbcgt_6,901). happensAt(walking(lbbcgt_6),901).
orientation(lbbcgt_3,163,902). coords(lbbcgt_3,39,152,902).  visible(lbbcgt_3,902). happensAt(walking(lbbcgt_3),902). orientation(lbbcgt_6,142,902). coords(lbbcgt_6,123,71,902).  visible(lbbcgt_6,902). happensAt(walking(lbbcgt_6),902).
orientation(lbbcgt_3,163,903). coords(lbbcgt_3,39,153,903).  visible(lbbcgt_3,903). happensAt(walking(lbbcgt_3),903). orientation(lbbcgt_6,142,903). coords(lbbcgt_6,122,71,903).  visible(lbbcgt_6,903). happensAt(walking(lbbcgt_6),903).
orientation(lbbcgt_3,163,904). coords(lbbcgt_3,38,155,904).  visible(lbbcgt_3,904). happensAt(walking(lbbcgt_3),904). orientation(lbbcgt_6,142,904). coords(lbbcgt_6,121,71,904).  visible(lbbcgt_6,904). happensAt(walking(lbbcgt_6),904).
orientation(lbbcgt_3,163,905). coords(lbbcgt_3,38,157,905).  visible(lbbcgt_3,905). happensAt(walking(lbbcgt_3),905). orientation(lbbcgt_6,142,905). coords(lbbcgt_6,121,72,905).  visible(lbbcgt_6,905). happensAt(walking(lbbcgt_6),905).
#show holdsAt/2.