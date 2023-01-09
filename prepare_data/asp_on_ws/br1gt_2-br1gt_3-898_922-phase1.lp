
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

	orientation(br1gt_2,2,898). coords(br1gt_2,85,201,898).  visible(br1gt_2,898). happensAt(active(br1gt_2),898). orientation(br1gt_3,94,898). coords(br1gt_3,288,131,898).  visible(br1gt_3,898). happensAt(walking(br1gt_3),898).
orientation(br1gt_2,2,899). coords(br1gt_2,85,200,899).  visible(br1gt_2,899). happensAt(active(br1gt_2),899). orientation(br1gt_3,94,899). coords(br1gt_3,289,129,899).  visible(br1gt_3,899). happensAt(walking(br1gt_3),899).
orientation(br1gt_2,2,900). coords(br1gt_2,85,200,900).  visible(br1gt_2,900). happensAt(active(br1gt_2),900). orientation(br1gt_3,94,900). coords(br1gt_3,289,126,900).  visible(br1gt_3,900). happensAt(walking(br1gt_3),900).
orientation(br1gt_2,2,901). coords(br1gt_2,85,200,901).  visible(br1gt_2,901). happensAt(active(br1gt_2),901). orientation(br1gt_3,94,901). coords(br1gt_3,289,124,901).  visible(br1gt_3,901). happensAt(walking(br1gt_3),901).
orientation(br1gt_2,2,902). coords(br1gt_2,85,200,902).  visible(br1gt_2,902). happensAt(active(br1gt_2),902). orientation(br1gt_3,94,902). coords(br1gt_3,290,123,902).  visible(br1gt_3,902). happensAt(walking(br1gt_3),902).
orientation(br1gt_2,2,903). coords(br1gt_2,85,200,903).  visible(br1gt_2,903). happensAt(active(br1gt_2),903). orientation(br1gt_3,94,903). coords(br1gt_3,290,119,903).  visible(br1gt_3,903). happensAt(walking(br1gt_3),903).
orientation(br1gt_2,2,904). coords(br1gt_2,85,200,904).  visible(br1gt_2,904). happensAt(active(br1gt_2),904). orientation(br1gt_3,89,904). coords(br1gt_3,290,117,904).  visible(br1gt_3,904). happensAt(walking(br1gt_3),904).
orientation(br1gt_2,2,905). coords(br1gt_2,85,200,905).  visible(br1gt_2,905). happensAt(active(br1gt_2),905). orientation(br1gt_3,89,905). coords(br1gt_3,290,115,905).  visible(br1gt_3,905). happensAt(walking(br1gt_3),905).
orientation(br1gt_2,2,906). coords(br1gt_2,85,200,906).  visible(br1gt_2,906). happensAt(active(br1gt_2),906). orientation(br1gt_3,89,906). coords(br1gt_3,291,113,906).  visible(br1gt_3,906). happensAt(walking(br1gt_3),906).
orientation(br1gt_2,2,907). coords(br1gt_2,85,200,907).  visible(br1gt_2,907). happensAt(active(br1gt_2),907). orientation(br1gt_3,89,907). coords(br1gt_3,291,113,907).  visible(br1gt_3,907). happensAt(walking(br1gt_3),907).
orientation(br1gt_2,2,908). coords(br1gt_2,85,200,908).  visible(br1gt_2,908). happensAt(active(br1gt_2),908). orientation(br1gt_3,89,908). coords(br1gt_3,291,112,908).  visible(br1gt_3,908). happensAt(walking(br1gt_3),908).
orientation(br1gt_2,2,909). coords(br1gt_2,85,200,909).  visible(br1gt_2,909). happensAt(active(br1gt_2),909). orientation(br1gt_3,89,909). coords(br1gt_3,294,112,909).  visible(br1gt_3,909). happensAt(walking(br1gt_3),909).
orientation(br1gt_2,2,910). coords(br1gt_2,85,200,910).  visible(br1gt_2,910). happensAt(active(br1gt_2),910). orientation(br1gt_3,89,910). coords(br1gt_3,292,111,910).  visible(br1gt_3,910). happensAt(walking(br1gt_3),910).
orientation(br1gt_2,2,911). coords(br1gt_2,85,200,911).  visible(br1gt_2,911). happensAt(active(br1gt_2),911). orientation(br1gt_3,89,911). coords(br1gt_3,294,110,911).  visible(br1gt_3,911). happensAt(walking(br1gt_3),911).
orientation(br1gt_2,2,912). coords(br1gt_2,85,200,912).  visible(br1gt_2,912). happensAt(active(br1gt_2),912). orientation(br1gt_3,89,912). coords(br1gt_3,294,110,912).  visible(br1gt_3,912). happensAt(walking(br1gt_3),912).
orientation(br1gt_2,2,913). coords(br1gt_2,85,199,913).  visible(br1gt_2,913). happensAt(active(br1gt_2),913). orientation(br1gt_3,89,913). coords(br1gt_3,294,109,913).  visible(br1gt_3,913). happensAt(walking(br1gt_3),913).
orientation(br1gt_2,2,914). coords(br1gt_2,85,199,914).  visible(br1gt_2,914). happensAt(active(br1gt_2),914). orientation(br1gt_3,89,914). coords(br1gt_3,295,109,914).  visible(br1gt_3,914). happensAt(walking(br1gt_3),914).
orientation(br1gt_2,2,915). coords(br1gt_2,85,199,915).  visible(br1gt_2,915). happensAt(active(br1gt_2),915). orientation(br1gt_3,89,915). coords(br1gt_3,296,108,915).  visible(br1gt_3,915). happensAt(walking(br1gt_3),915).
orientation(br1gt_2,2,916). coords(br1gt_2,85,199,916).  visible(br1gt_2,916). happensAt(active(br1gt_2),916). orientation(br1gt_3,89,916). coords(br1gt_3,295,107,916).  visible(br1gt_3,916). happensAt(walking(br1gt_3),916).
orientation(br1gt_2,2,917). coords(br1gt_2,85,199,917).  visible(br1gt_2,917). happensAt(active(br1gt_2),917). orientation(br1gt_3,89,917). coords(br1gt_3,295,106,917).  visible(br1gt_3,917). happensAt(walking(br1gt_3),917).
orientation(br1gt_2,2,918). coords(br1gt_2,85,199,918).  visible(br1gt_2,918). happensAt(active(br1gt_2),918). orientation(br1gt_3,89,918). coords(br1gt_3,295,105,918).  visible(br1gt_3,918). happensAt(walking(br1gt_3),918).
orientation(br1gt_2,2,919). coords(br1gt_2,85,199,919).  visible(br1gt_2,919). happensAt(active(br1gt_2),919). orientation(br1gt_3,89,919). coords(br1gt_3,295,103,919).  visible(br1gt_3,919). happensAt(walking(br1gt_3),919).
orientation(br1gt_2,2,920). coords(br1gt_2,85,199,920).  visible(br1gt_2,920). happensAt(active(br1gt_2),920). orientation(br1gt_3,89,920). coords(br1gt_3,296,101,920).  visible(br1gt_3,920). happensAt(walking(br1gt_3),920).
orientation(br1gt_2,2,921). coords(br1gt_2,86,197,921).  visible(br1gt_2,921). happensAt(active(br1gt_2),921). orientation(br1gt_3,89,921). coords(br1gt_3,297,99,921).  visible(br1gt_3,921). happensAt(walking(br1gt_3),921).
orientation(br1gt_2,2,922). coords(br1gt_2,86,198,922).  visible(br1gt_2,922). happensAt(active(br1gt_2),922). orientation(br1gt_3,89,922). coords(br1gt_3,297,98,922).  visible(br1gt_3,922). happensAt(active(br1gt_3),922).
#show holdsAt/2.