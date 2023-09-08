
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

	orientation(fra2gt_3,146,526). coords(fra2gt_3,44,116,526).  visible(fra2gt_3,526). happensAt(walking(fra2gt_3),526). orientation(fra2gt_4,146,526). coords(fra2gt_4,64,89,526).  visible(fra2gt_4,526). happensAt(walking(fra2gt_4),526).
orientation(fra2gt_3,146,527). coords(fra2gt_3,44,114,527).  visible(fra2gt_3,527). happensAt(walking(fra2gt_3),527). orientation(fra2gt_4,146,527). coords(fra2gt_4,64,89,527).  visible(fra2gt_4,527). happensAt(walking(fra2gt_4),527).
orientation(fra2gt_3,146,528). coords(fra2gt_3,44,114,528).  visible(fra2gt_3,528). happensAt(walking(fra2gt_3),528). orientation(fra2gt_4,146,528). coords(fra2gt_4,64,89,528).  visible(fra2gt_4,528). happensAt(walking(fra2gt_4),528).
orientation(fra2gt_3,146,529). coords(fra2gt_3,44,113,529).  visible(fra2gt_3,529). happensAt(walking(fra2gt_3),529). orientation(fra2gt_4,146,529). coords(fra2gt_4,65,90,529).  visible(fra2gt_4,529). happensAt(walking(fra2gt_4),529).
orientation(fra2gt_3,146,530). coords(fra2gt_3,44,113,530).  visible(fra2gt_3,530). happensAt(walking(fra2gt_3),530). orientation(fra2gt_4,146,530). coords(fra2gt_4,65,89,530).  visible(fra2gt_4,530). happensAt(walking(fra2gt_4),530).
orientation(fra2gt_3,146,531). coords(fra2gt_3,44,108,531).  visible(fra2gt_3,531). happensAt(walking(fra2gt_3),531). orientation(fra2gt_4,146,531). coords(fra2gt_4,65,89,531).  visible(fra2gt_4,531). happensAt(walking(fra2gt_4),531).
orientation(fra2gt_3,146,532). coords(fra2gt_3,44,109,532).  visible(fra2gt_3,532). happensAt(walking(fra2gt_3),532). orientation(fra2gt_4,146,532). coords(fra2gt_4,65,90,532).  visible(fra2gt_4,532). happensAt(walking(fra2gt_4),532).
orientation(fra2gt_3,146,533). coords(fra2gt_3,44,109,533).  visible(fra2gt_3,533). happensAt(walking(fra2gt_3),533). orientation(fra2gt_4,146,533). coords(fra2gt_4,65,90,533).  visible(fra2gt_4,533). happensAt(walking(fra2gt_4),533).
orientation(fra2gt_3,146,534). coords(fra2gt_3,44,107,534).  visible(fra2gt_3,534). happensAt(walking(fra2gt_3),534). orientation(fra2gt_4,146,534). coords(fra2gt_4,65,90,534).  visible(fra2gt_4,534). happensAt(walking(fra2gt_4),534).
orientation(fra2gt_3,146,535). coords(fra2gt_3,45,106,535).  visible(fra2gt_3,535). happensAt(walking(fra2gt_3),535). orientation(fra2gt_4,146,535). coords(fra2gt_4,66,91,535).  visible(fra2gt_4,535). happensAt(walking(fra2gt_4),535).
orientation(fra2gt_3,146,536). coords(fra2gt_3,46,106,536).  visible(fra2gt_3,536). happensAt(walking(fra2gt_3),536). orientation(fra2gt_4,146,536). coords(fra2gt_4,66,91,536).  visible(fra2gt_4,536). happensAt(walking(fra2gt_4),536).
orientation(fra2gt_3,146,537). coords(fra2gt_3,46,104,537).  visible(fra2gt_3,537). happensAt(walking(fra2gt_3),537). orientation(fra2gt_4,146,537). coords(fra2gt_4,67,91,537).  visible(fra2gt_4,537). happensAt(walking(fra2gt_4),537).
orientation(fra2gt_3,146,538). coords(fra2gt_3,46,103,538).  visible(fra2gt_3,538). happensAt(walking(fra2gt_3),538). orientation(fra2gt_4,146,538). coords(fra2gt_4,67,91,538).  visible(fra2gt_4,538). happensAt(walking(fra2gt_4),538).
orientation(fra2gt_3,146,539). coords(fra2gt_3,46,103,539).  visible(fra2gt_3,539). happensAt(walking(fra2gt_3),539). orientation(fra2gt_4,146,539). coords(fra2gt_4,67,92,539).  visible(fra2gt_4,539). happensAt(walking(fra2gt_4),539).
orientation(fra2gt_3,146,540). coords(fra2gt_3,48,103,540).  visible(fra2gt_3,540). happensAt(walking(fra2gt_3),540). orientation(fra2gt_4,146,540). coords(fra2gt_4,67,92,540).  visible(fra2gt_4,540). happensAt(walking(fra2gt_4),540).
orientation(fra2gt_3,146,541). coords(fra2gt_3,48,101,541).  visible(fra2gt_3,541). happensAt(walking(fra2gt_3),541). orientation(fra2gt_4,146,541). coords(fra2gt_4,67,92,541).  visible(fra2gt_4,541). happensAt(walking(fra2gt_4),541).
orientation(fra2gt_3,146,542). coords(fra2gt_3,49,101,542).  visible(fra2gt_3,542). happensAt(walking(fra2gt_3),542). orientation(fra2gt_4,146,542). coords(fra2gt_4,67,92,542).  visible(fra2gt_4,542). happensAt(walking(fra2gt_4),542).
orientation(fra2gt_3,146,543). coords(fra2gt_3,51,101,543).  visible(fra2gt_3,543). happensAt(walking(fra2gt_3),543). orientation(fra2gt_4,146,543). coords(fra2gt_4,67,92,543).  visible(fra2gt_4,543). happensAt(walking(fra2gt_4),543).
orientation(fra2gt_3,146,544). coords(fra2gt_3,50,101,544).  visible(fra2gt_3,544). happensAt(walking(fra2gt_3),544). orientation(fra2gt_4,146,544). coords(fra2gt_4,67,93,544).  visible(fra2gt_4,544). happensAt(walking(fra2gt_4),544).
orientation(fra2gt_3,146,545). coords(fra2gt_3,53,100,545).  visible(fra2gt_3,545). happensAt(walking(fra2gt_3),545). orientation(fra2gt_4,146,545). coords(fra2gt_4,67,94,545).  visible(fra2gt_4,545). happensAt(walking(fra2gt_4),545).
orientation(fra2gt_3,146,546). coords(fra2gt_3,54,100,546).  visible(fra2gt_3,546). happensAt(walking(fra2gt_3),546). orientation(fra2gt_4,146,546). coords(fra2gt_4,67,95,546).  visible(fra2gt_4,546). happensAt(walking(fra2gt_4),546).
orientation(fra2gt_3,146,547). coords(fra2gt_3,54,99,547).  visible(fra2gt_3,547). happensAt(walking(fra2gt_3),547). orientation(fra2gt_4,146,547). coords(fra2gt_4,67,95,547).  visible(fra2gt_4,547). happensAt(walking(fra2gt_4),547).
orientation(fra2gt_3,146,548). coords(fra2gt_3,54,99,548).  visible(fra2gt_3,548). happensAt(walking(fra2gt_3),548). orientation(fra2gt_4,146,548). coords(fra2gt_4,68,95,548).  visible(fra2gt_4,548). happensAt(walking(fra2gt_4),548).
orientation(fra2gt_3,146,549). coords(fra2gt_3,54,99,549).  visible(fra2gt_3,549). happensAt(walking(fra2gt_3),549). orientation(fra2gt_4,146,549). coords(fra2gt_4,68,95,549).  visible(fra2gt_4,549). happensAt(walking(fra2gt_4),549).
orientation(fra2gt_3,146,550). coords(fra2gt_3,55,99,550).  happensAt(disappear(fra2gt_3),550). happensAt(walking(fra2gt_3),550). orientation(fra2gt_4,146,550). coords(fra2gt_4,69,95,550).  happensAt(disappear(fra2gt_4),550). happensAt(walking(fra2gt_4),550).
#show holdsAt/2.