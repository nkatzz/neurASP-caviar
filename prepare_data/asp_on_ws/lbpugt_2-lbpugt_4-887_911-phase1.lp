
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

	orientation(lbpugt_2,134,887). coords(lbpugt_2,268,119,887).  visible(lbpugt_2,887). happensAt(walking(lbpugt_2),887). orientation(lbpugt_4,135,887). coords(lbpugt_4,219,34,887).  visible(lbpugt_4,887). happensAt(inactive(lbpugt_4),887).
orientation(lbpugt_2,141,888). coords(lbpugt_2,267,120,888).  visible(lbpugt_2,888). happensAt(walking(lbpugt_2),888). orientation(lbpugt_4,135,888). coords(lbpugt_4,219,34,888).  visible(lbpugt_4,888). happensAt(inactive(lbpugt_4),888).
orientation(lbpugt_2,141,889). coords(lbpugt_2,265,121,889).  visible(lbpugt_2,889). happensAt(walking(lbpugt_2),889). orientation(lbpugt_4,135,889). coords(lbpugt_4,219,34,889).  visible(lbpugt_4,889). happensAt(inactive(lbpugt_4),889).
orientation(lbpugt_2,146,890). coords(lbpugt_2,263,121,890).  visible(lbpugt_2,890). happensAt(walking(lbpugt_2),890). orientation(lbpugt_4,135,890). coords(lbpugt_4,219,34,890).  visible(lbpugt_4,890). happensAt(inactive(lbpugt_4),890).
orientation(lbpugt_2,146,891). coords(lbpugt_2,262,121,891).  visible(lbpugt_2,891). happensAt(walking(lbpugt_2),891). orientation(lbpugt_4,135,891). coords(lbpugt_4,219,34,891).  visible(lbpugt_4,891). happensAt(inactive(lbpugt_4),891).
orientation(lbpugt_2,146,892). coords(lbpugt_2,259,122,892).  visible(lbpugt_2,892). happensAt(walking(lbpugt_2),892). orientation(lbpugt_4,135,892). coords(lbpugt_4,219,34,892).  visible(lbpugt_4,892). happensAt(inactive(lbpugt_4),892).
orientation(lbpugt_2,146,893). coords(lbpugt_2,257,122,893).  visible(lbpugt_2,893). happensAt(walking(lbpugt_2),893). orientation(lbpugt_4,135,893). coords(lbpugt_4,219,34,893).  visible(lbpugt_4,893). happensAt(inactive(lbpugt_4),893).
orientation(lbpugt_2,142,894). coords(lbpugt_2,254,122,894).  visible(lbpugt_2,894). happensAt(walking(lbpugt_2),894). orientation(lbpugt_4,135,894). coords(lbpugt_4,219,34,894).  visible(lbpugt_4,894). happensAt(inactive(lbpugt_4),894).
orientation(lbpugt_2,136,895). coords(lbpugt_2,252,123,895).  visible(lbpugt_2,895). happensAt(walking(lbpugt_2),895). orientation(lbpugt_4,135,895). coords(lbpugt_4,219,34,895).  visible(lbpugt_4,895). happensAt(inactive(lbpugt_4),895).
orientation(lbpugt_2,136,896). coords(lbpugt_2,251,123,896).  visible(lbpugt_2,896). happensAt(walking(lbpugt_2),896). orientation(lbpugt_4,135,896). coords(lbpugt_4,219,34,896).  visible(lbpugt_4,896). happensAt(inactive(lbpugt_4),896).
orientation(lbpugt_2,136,897). coords(lbpugt_2,250,124,897).  visible(lbpugt_2,897). happensAt(walking(lbpugt_2),897). orientation(lbpugt_4,135,897). coords(lbpugt_4,219,34,897).  visible(lbpugt_4,897). happensAt(inactive(lbpugt_4),897).
orientation(lbpugt_2,136,898). coords(lbpugt_2,250,124,898).  visible(lbpugt_2,898). happensAt(walking(lbpugt_2),898). orientation(lbpugt_4,135,898). coords(lbpugt_4,219,34,898).  visible(lbpugt_4,898). happensAt(inactive(lbpugt_4),898).
orientation(lbpugt_2,143,899). coords(lbpugt_2,249,124,899).  visible(lbpugt_2,899). happensAt(walking(lbpugt_2),899). orientation(lbpugt_4,135,899). coords(lbpugt_4,219,34,899).  visible(lbpugt_4,899). happensAt(inactive(lbpugt_4),899).
orientation(lbpugt_2,143,900). coords(lbpugt_2,248,125,900).  visible(lbpugt_2,900). happensAt(walking(lbpugt_2),900). orientation(lbpugt_4,135,900). coords(lbpugt_4,219,34,900).  visible(lbpugt_4,900). happensAt(inactive(lbpugt_4),900).
orientation(lbpugt_2,143,901). coords(lbpugt_2,247,125,901).  visible(lbpugt_2,901). happensAt(walking(lbpugt_2),901). orientation(lbpugt_4,135,901). coords(lbpugt_4,219,34,901).  visible(lbpugt_4,901). happensAt(inactive(lbpugt_4),901).
orientation(lbpugt_2,143,902). coords(lbpugt_2,246,125,902).  visible(lbpugt_2,902). happensAt(walking(lbpugt_2),902). orientation(lbpugt_4,135,902). coords(lbpugt_4,219,34,902).  visible(lbpugt_4,902). happensAt(inactive(lbpugt_4),902).
orientation(lbpugt_2,143,903). coords(lbpugt_2,245,124,903).  visible(lbpugt_2,903). happensAt(walking(lbpugt_2),903). orientation(lbpugt_4,135,903). coords(lbpugt_4,219,34,903).  visible(lbpugt_4,903). happensAt(inactive(lbpugt_4),903).
orientation(lbpugt_2,143,904). coords(lbpugt_2,243,124,904).  visible(lbpugt_2,904). happensAt(walking(lbpugt_2),904). orientation(lbpugt_4,135,904). coords(lbpugt_4,219,34,904).  visible(lbpugt_4,904). happensAt(inactive(lbpugt_4),904).
orientation(lbpugt_2,143,905). coords(lbpugt_2,242,123,905).  visible(lbpugt_2,905). happensAt(walking(lbpugt_2),905). orientation(lbpugt_4,135,905). coords(lbpugt_4,219,34,905).  visible(lbpugt_4,905). happensAt(inactive(lbpugt_4),905).
orientation(lbpugt_2,137,906). coords(lbpugt_2,238,123,906).  visible(lbpugt_2,906). happensAt(walking(lbpugt_2),906). orientation(lbpugt_4,135,906). coords(lbpugt_4,219,34,906).  visible(lbpugt_4,906). happensAt(inactive(lbpugt_4),906).
orientation(lbpugt_2,137,907). coords(lbpugt_2,236,122,907).  visible(lbpugt_2,907). happensAt(walking(lbpugt_2),907). orientation(lbpugt_4,135,907). coords(lbpugt_4,219,34,907).  visible(lbpugt_4,907). happensAt(inactive(lbpugt_4),907).
orientation(lbpugt_2,131,908). coords(lbpugt_2,234,122,908).  visible(lbpugt_2,908). happensAt(walking(lbpugt_2),908). orientation(lbpugt_4,135,908). coords(lbpugt_4,219,34,908).  visible(lbpugt_4,908). happensAt(inactive(lbpugt_4),908).
orientation(lbpugt_2,126,909). coords(lbpugt_2,230,121,909).  visible(lbpugt_2,909). happensAt(walking(lbpugt_2),909). orientation(lbpugt_4,135,909). coords(lbpugt_4,219,34,909).  visible(lbpugt_4,909). happensAt(inactive(lbpugt_4),909).
orientation(lbpugt_2,125,910). coords(lbpugt_2,227,120,910).  visible(lbpugt_2,910). happensAt(walking(lbpugt_2),910). orientation(lbpugt_4,135,910). coords(lbpugt_4,219,34,910).  visible(lbpugt_4,910). happensAt(inactive(lbpugt_4),910).
orientation(lbpugt_2,125,911). coords(lbpugt_2,227,120,911).  visible(lbpugt_2,911). happensAt(walking(lbpugt_2),911). orientation(lbpugt_4,135,911). coords(lbpugt_4,219,34,911).  visible(lbpugt_4,911). happensAt(inactive(lbpugt_4),911).
#show holdsAt/2.