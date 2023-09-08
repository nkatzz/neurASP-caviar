
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

	orientation(lbbcgt_5,20,885). coords(lbbcgt_5,241,280,885).  visible(lbbcgt_5,885). happensAt(walking(lbbcgt_5),885). orientation(lbbcgt_6,142,885). coords(lbbcgt_6,145,56,885).  visible(lbbcgt_6,885). happensAt(walking(lbbcgt_6),885).
orientation(lbbcgt_5,20,886). coords(lbbcgt_5,239,278,886).  visible(lbbcgt_5,886). happensAt(walking(lbbcgt_5),886). orientation(lbbcgt_6,142,886). coords(lbbcgt_6,143,57,886).  visible(lbbcgt_6,886). happensAt(walking(lbbcgt_6),886).
orientation(lbbcgt_5,20,887). coords(lbbcgt_5,238,274,887).  visible(lbbcgt_5,887). happensAt(walking(lbbcgt_5),887). orientation(lbbcgt_6,142,887). coords(lbbcgt_6,142,58,887).  visible(lbbcgt_6,887). happensAt(walking(lbbcgt_6),887).
orientation(lbbcgt_5,20,888). coords(lbbcgt_5,237,271,888).  visible(lbbcgt_5,888). happensAt(walking(lbbcgt_5),888). orientation(lbbcgt_6,142,888). coords(lbbcgt_6,140,58,888).  visible(lbbcgt_6,888). happensAt(walking(lbbcgt_6),888).
orientation(lbbcgt_5,20,889). coords(lbbcgt_5,236,268,889).  visible(lbbcgt_5,889). happensAt(walking(lbbcgt_5),889). orientation(lbbcgt_6,142,889). coords(lbbcgt_6,138,58,889).  visible(lbbcgt_6,889). happensAt(walking(lbbcgt_6),889).
orientation(lbbcgt_5,20,890). coords(lbbcgt_5,236,267,890).  visible(lbbcgt_5,890). happensAt(walking(lbbcgt_5),890). orientation(lbbcgt_6,142,890). coords(lbbcgt_6,136,59,890).  visible(lbbcgt_6,890). happensAt(walking(lbbcgt_6),890).
orientation(lbbcgt_5,21,891). coords(lbbcgt_5,236,266,891).  visible(lbbcgt_5,891). happensAt(walking(lbbcgt_5),891). orientation(lbbcgt_6,142,891). coords(lbbcgt_6,135,60,891).  visible(lbbcgt_6,891). happensAt(walking(lbbcgt_6),891).
orientation(lbbcgt_5,21,892). coords(lbbcgt_5,236,266,892).  visible(lbbcgt_5,892). happensAt(walking(lbbcgt_5),892). orientation(lbbcgt_6,142,892). coords(lbbcgt_6,134,63,892).  visible(lbbcgt_6,892). happensAt(walking(lbbcgt_6),892).
orientation(lbbcgt_5,21,893). coords(lbbcgt_5,235,265,893).  visible(lbbcgt_5,893). happensAt(walking(lbbcgt_5),893). orientation(lbbcgt_6,142,893). coords(lbbcgt_6,134,64,893).  visible(lbbcgt_6,893). happensAt(walking(lbbcgt_6),893).
orientation(lbbcgt_5,21,894). coords(lbbcgt_5,234,264,894).  visible(lbbcgt_5,894). happensAt(walking(lbbcgt_5),894). orientation(lbbcgt_6,142,894). coords(lbbcgt_6,133,66,894).  visible(lbbcgt_6,894). happensAt(walking(lbbcgt_6),894).
orientation(lbbcgt_5,21,895). coords(lbbcgt_5,234,263,895).  visible(lbbcgt_5,895). happensAt(walking(lbbcgt_5),895). orientation(lbbcgt_6,142,895). coords(lbbcgt_6,133,67,895).  visible(lbbcgt_6,895). happensAt(walking(lbbcgt_6),895).
orientation(lbbcgt_5,21,896). coords(lbbcgt_5,231,261,896).  visible(lbbcgt_5,896). happensAt(walking(lbbcgt_5),896). orientation(lbbcgt_6,142,896). coords(lbbcgt_6,132,68,896).  visible(lbbcgt_6,896). happensAt(walking(lbbcgt_6),896).
orientation(lbbcgt_5,21,897). coords(lbbcgt_5,231,259,897).  visible(lbbcgt_5,897). happensAt(walking(lbbcgt_5),897). orientation(lbbcgt_6,142,897). coords(lbbcgt_6,132,68,897).  visible(lbbcgt_6,897). happensAt(walking(lbbcgt_6),897).
orientation(lbbcgt_5,9,898). coords(lbbcgt_5,230,257,898).  visible(lbbcgt_5,898). happensAt(walking(lbbcgt_5),898). orientation(lbbcgt_6,142,898). coords(lbbcgt_6,131,69,898).  visible(lbbcgt_6,898). happensAt(walking(lbbcgt_6),898).
orientation(lbbcgt_5,9,899). coords(lbbcgt_5,229,256,899).  visible(lbbcgt_5,899). happensAt(walking(lbbcgt_5),899). orientation(lbbcgt_6,142,899). coords(lbbcgt_6,129,70,899).  visible(lbbcgt_6,899). happensAt(walking(lbbcgt_6),899).
orientation(lbbcgt_5,9,900). coords(lbbcgt_5,229,255,900).  visible(lbbcgt_5,900). happensAt(walking(lbbcgt_5),900). orientation(lbbcgt_6,142,900). coords(lbbcgt_6,127,70,900).  visible(lbbcgt_6,900). happensAt(walking(lbbcgt_6),900).
orientation(lbbcgt_5,9,901). coords(lbbcgt_5,229,253,901).  visible(lbbcgt_5,901). happensAt(walking(lbbcgt_5),901). orientation(lbbcgt_6,142,901). coords(lbbcgt_6,125,70,901).  visible(lbbcgt_6,901). happensAt(walking(lbbcgt_6),901).
orientation(lbbcgt_5,9,902). coords(lbbcgt_5,228,251,902).  visible(lbbcgt_5,902). happensAt(walking(lbbcgt_5),902). orientation(lbbcgt_6,142,902). coords(lbbcgt_6,123,71,902).  visible(lbbcgt_6,902). happensAt(walking(lbbcgt_6),902).
orientation(lbbcgt_5,9,903). coords(lbbcgt_5,228,247,903).  visible(lbbcgt_5,903). happensAt(walking(lbbcgt_5),903). orientation(lbbcgt_6,142,903). coords(lbbcgt_6,122,71,903).  visible(lbbcgt_6,903). happensAt(walking(lbbcgt_6),903).
orientation(lbbcgt_5,9,904). coords(lbbcgt_5,227,244,904).  visible(lbbcgt_5,904). happensAt(walking(lbbcgt_5),904). orientation(lbbcgt_6,142,904). coords(lbbcgt_6,121,71,904).  visible(lbbcgt_6,904). happensAt(walking(lbbcgt_6),904).
orientation(lbbcgt_5,9,905). coords(lbbcgt_5,227,242,905).  visible(lbbcgt_5,905). happensAt(walking(lbbcgt_5),905). orientation(lbbcgt_6,142,905). coords(lbbcgt_6,121,72,905).  visible(lbbcgt_6,905). happensAt(walking(lbbcgt_6),905).
orientation(lbbcgt_5,9,906). coords(lbbcgt_5,226,240,906).  visible(lbbcgt_5,906). happensAt(walking(lbbcgt_5),906). orientation(lbbcgt_6,142,906). coords(lbbcgt_6,120,73,906).  visible(lbbcgt_6,906). happensAt(walking(lbbcgt_6),906).
orientation(lbbcgt_5,9,907). coords(lbbcgt_5,225,240,907).  visible(lbbcgt_5,907). happensAt(walking(lbbcgt_5),907). orientation(lbbcgt_6,142,907). coords(lbbcgt_6,120,73,907).  visible(lbbcgt_6,907). happensAt(walking(lbbcgt_6),907).
orientation(lbbcgt_5,9,908). coords(lbbcgt_5,225,240,908).  visible(lbbcgt_5,908). happensAt(walking(lbbcgt_5),908). orientation(lbbcgt_6,142,908). coords(lbbcgt_6,120,74,908).  visible(lbbcgt_6,908). happensAt(walking(lbbcgt_6),908).
orientation(lbbcgt_5,9,909). coords(lbbcgt_5,224,239,909).  visible(lbbcgt_5,909). happensAt(walking(lbbcgt_5),909). orientation(lbbcgt_6,142,909). coords(lbbcgt_6,118,74,909).  visible(lbbcgt_6,909). happensAt(walking(lbbcgt_6),909).
#show holdsAt/2.