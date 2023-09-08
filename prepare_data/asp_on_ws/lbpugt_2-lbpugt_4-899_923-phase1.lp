
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
orientation(lbpugt_2,130,912). coords(lbpugt_2,225,119,912).  visible(lbpugt_2,912). happensAt(walking(lbpugt_2),912). orientation(lbpugt_4,135,912). coords(lbpugt_4,219,34,912).  visible(lbpugt_4,912). happensAt(inactive(lbpugt_4),912).
orientation(lbpugt_2,130,913). coords(lbpugt_2,225,118,913).  visible(lbpugt_2,913). happensAt(walking(lbpugt_2),913). orientation(lbpugt_4,135,913). coords(lbpugt_4,219,34,913).  visible(lbpugt_4,913). happensAt(inactive(lbpugt_4),913).
orientation(lbpugt_2,130,914). coords(lbpugt_2,224,117,914).  visible(lbpugt_2,914). happensAt(walking(lbpugt_2),914). orientation(lbpugt_4,135,914). coords(lbpugt_4,219,34,914).  visible(lbpugt_4,914). happensAt(inactive(lbpugt_4),914).
orientation(lbpugt_2,130,915). coords(lbpugt_2,223,117,915).  visible(lbpugt_2,915). happensAt(walking(lbpugt_2),915). orientation(lbpugt_4,135,915). coords(lbpugt_4,219,34,915).  visible(lbpugt_4,915). happensAt(inactive(lbpugt_4),915).
orientation(lbpugt_2,130,916). coords(lbpugt_2,222,117,916).  visible(lbpugt_2,916). happensAt(walking(lbpugt_2),916). orientation(lbpugt_4,135,916). coords(lbpugt_4,219,34,916).  visible(lbpugt_4,916). happensAt(inactive(lbpugt_4),916).
orientation(lbpugt_2,134,917). coords(lbpugt_2,221,117,917).  visible(lbpugt_2,917). happensAt(walking(lbpugt_2),917). orientation(lbpugt_4,135,917). coords(lbpugt_4,219,34,917).  visible(lbpugt_4,917). happensAt(inactive(lbpugt_4),917).
orientation(lbpugt_2,134,918). coords(lbpugt_2,220,116,918).  visible(lbpugt_2,918). happensAt(walking(lbpugt_2),918). orientation(lbpugt_4,135,918). coords(lbpugt_4,219,34,918).  visible(lbpugt_4,918). happensAt(inactive(lbpugt_4),918).
orientation(lbpugt_2,137,919). coords(lbpugt_2,219,115,919).  visible(lbpugt_2,919). happensAt(walking(lbpugt_2),919). orientation(lbpugt_4,135,919). coords(lbpugt_4,219,34,919).  visible(lbpugt_4,919). happensAt(inactive(lbpugt_4),919).
orientation(lbpugt_2,137,920). coords(lbpugt_2,217,114,920).  visible(lbpugt_2,920). happensAt(walking(lbpugt_2),920). orientation(lbpugt_4,135,920). coords(lbpugt_4,219,34,920).  visible(lbpugt_4,920). happensAt(inactive(lbpugt_4),920).
orientation(lbpugt_2,137,921). coords(lbpugt_2,215,112,921).  visible(lbpugt_2,921). happensAt(walking(lbpugt_2),921). orientation(lbpugt_4,135,921). coords(lbpugt_4,219,34,921).  visible(lbpugt_4,921). happensAt(inactive(lbpugt_4),921).
orientation(lbpugt_2,137,922). coords(lbpugt_2,213,110,922).  visible(lbpugt_2,922). happensAt(walking(lbpugt_2),922). orientation(lbpugt_4,135,922). coords(lbpugt_4,219,34,922).  visible(lbpugt_4,922). happensAt(inactive(lbpugt_4),922).
orientation(lbpugt_2,137,923). coords(lbpugt_2,211,108,923).  visible(lbpugt_2,923). happensAt(walking(lbpugt_2),923). orientation(lbpugt_4,135,923). coords(lbpugt_4,219,34,923).  visible(lbpugt_4,923). happensAt(inactive(lbpugt_4),923).
#show holdsAt/2.