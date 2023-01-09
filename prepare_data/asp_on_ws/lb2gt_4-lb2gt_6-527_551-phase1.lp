
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

	orientation(lb2gt_4,154,527). coords(lb2gt_4,93,74,527).  visible(lb2gt_4,527). happensAt(walking(lb2gt_4),527). orientation(lb2gt_6,151,527). coords(lb2gt_6,72,122,527).  visible(lb2gt_6,527). happensAt(inactive(lb2gt_6),527).
orientation(lb2gt_4,154,528). coords(lb2gt_4,94,74,528).  visible(lb2gt_4,528). happensAt(walking(lb2gt_4),528). orientation(lb2gt_6,151,528). coords(lb2gt_6,72,122,528).  visible(lb2gt_6,528). happensAt(inactive(lb2gt_6),528).
orientation(lb2gt_4,154,529). coords(lb2gt_4,96,73,529).  visible(lb2gt_4,529). happensAt(walking(lb2gt_4),529). orientation(lb2gt_6,151,529). coords(lb2gt_6,72,122,529).  visible(lb2gt_6,529). happensAt(inactive(lb2gt_6),529).
orientation(lb2gt_4,154,530). coords(lb2gt_4,97,73,530).  visible(lb2gt_4,530). happensAt(walking(lb2gt_4),530). orientation(lb2gt_6,151,530). coords(lb2gt_6,72,122,530).  visible(lb2gt_6,530). happensAt(inactive(lb2gt_6),530).
orientation(lb2gt_4,154,531). coords(lb2gt_4,99,72,531).  visible(lb2gt_4,531). happensAt(walking(lb2gt_4),531). orientation(lb2gt_6,151,531). coords(lb2gt_6,72,122,531).  visible(lb2gt_6,531). happensAt(inactive(lb2gt_6),531).
orientation(lb2gt_4,154,532). coords(lb2gt_4,100,72,532).  visible(lb2gt_4,532). happensAt(walking(lb2gt_4),532). orientation(lb2gt_6,151,532). coords(lb2gt_6,72,122,532).  visible(lb2gt_6,532). happensAt(inactive(lb2gt_6),532).
orientation(lb2gt_4,154,533). coords(lb2gt_4,100,71,533).  visible(lb2gt_4,533). happensAt(walking(lb2gt_4),533). orientation(lb2gt_6,151,533). coords(lb2gt_6,72,122,533).  visible(lb2gt_6,533). happensAt(inactive(lb2gt_6),533).
orientation(lb2gt_4,154,534). coords(lb2gt_4,102,70,534).  visible(lb2gt_4,534). happensAt(walking(lb2gt_4),534). orientation(lb2gt_6,151,534). coords(lb2gt_6,72,122,534).  visible(lb2gt_6,534). happensAt(inactive(lb2gt_6),534).
orientation(lb2gt_4,154,535). coords(lb2gt_4,102,68,535).  visible(lb2gt_4,535). happensAt(walking(lb2gt_4),535). orientation(lb2gt_6,151,535). coords(lb2gt_6,72,122,535).  visible(lb2gt_6,535). happensAt(inactive(lb2gt_6),535).
orientation(lb2gt_4,154,536). coords(lb2gt_4,103,67,536).  visible(lb2gt_4,536). happensAt(walking(lb2gt_4),536). orientation(lb2gt_6,151,536). coords(lb2gt_6,72,122,536).  visible(lb2gt_6,536). happensAt(inactive(lb2gt_6),536).
orientation(lb2gt_4,154,537). coords(lb2gt_4,103,66,537).  visible(lb2gt_4,537). happensAt(walking(lb2gt_4),537). orientation(lb2gt_6,151,537). coords(lb2gt_6,72,122,537).  visible(lb2gt_6,537). happensAt(inactive(lb2gt_6),537).
orientation(lb2gt_4,154,538). coords(lb2gt_4,103,64,538).  visible(lb2gt_4,538). happensAt(walking(lb2gt_4),538). orientation(lb2gt_6,151,538). coords(lb2gt_6,72,122,538).  visible(lb2gt_6,538). happensAt(inactive(lb2gt_6),538).
orientation(lb2gt_4,154,539). coords(lb2gt_4,103,62,539).  visible(lb2gt_4,539). happensAt(walking(lb2gt_4),539). orientation(lb2gt_6,151,539). coords(lb2gt_6,72,122,539).  visible(lb2gt_6,539). happensAt(inactive(lb2gt_6),539).
orientation(lb2gt_4,154,540). coords(lb2gt_4,104,60,540).  visible(lb2gt_4,540). happensAt(walking(lb2gt_4),540). orientation(lb2gt_6,151,540). coords(lb2gt_6,72,122,540).  visible(lb2gt_6,540). happensAt(inactive(lb2gt_6),540).
orientation(lb2gt_4,154,541). coords(lb2gt_4,104,60,541).  visible(lb2gt_4,541). happensAt(walking(lb2gt_4),541). orientation(lb2gt_6,151,541). coords(lb2gt_6,72,122,541).  visible(lb2gt_6,541). happensAt(inactive(lb2gt_6),541).
orientation(lb2gt_4,154,542). coords(lb2gt_4,104,58,542).  visible(lb2gt_4,542). happensAt(walking(lb2gt_4),542). orientation(lb2gt_6,151,542). coords(lb2gt_6,72,122,542).  visible(lb2gt_6,542). happensAt(inactive(lb2gt_6),542).
orientation(lb2gt_4,154,543). coords(lb2gt_4,104,58,543).  visible(lb2gt_4,543). happensAt(walking(lb2gt_4),543). orientation(lb2gt_6,151,543). coords(lb2gt_6,72,122,543).  visible(lb2gt_6,543). happensAt(inactive(lb2gt_6),543).
orientation(lb2gt_4,154,544). coords(lb2gt_4,105,57,544).  visible(lb2gt_4,544). happensAt(walking(lb2gt_4),544). orientation(lb2gt_6,151,544). coords(lb2gt_6,72,122,544).  visible(lb2gt_6,544). happensAt(inactive(lb2gt_6),544).
orientation(lb2gt_4,154,545). coords(lb2gt_4,107,56,545).  visible(lb2gt_4,545). happensAt(walking(lb2gt_4),545). orientation(lb2gt_6,151,545). coords(lb2gt_6,72,122,545).  visible(lb2gt_6,545). happensAt(inactive(lb2gt_6),545).
orientation(lb2gt_4,154,546). coords(lb2gt_4,108,56,546).  visible(lb2gt_4,546). happensAt(walking(lb2gt_4),546). orientation(lb2gt_6,151,546). coords(lb2gt_6,72,122,546).  visible(lb2gt_6,546). happensAt(inactive(lb2gt_6),546).
orientation(lb2gt_4,154,547). coords(lb2gt_4,110,56,547).  visible(lb2gt_4,547). happensAt(walking(lb2gt_4),547). orientation(lb2gt_6,151,547). coords(lb2gt_6,72,122,547).  visible(lb2gt_6,547). happensAt(inactive(lb2gt_6),547).
orientation(lb2gt_4,154,548). coords(lb2gt_4,111,56,548).  visible(lb2gt_4,548). happensAt(walking(lb2gt_4),548). orientation(lb2gt_6,151,548). coords(lb2gt_6,72,122,548).  visible(lb2gt_6,548). happensAt(inactive(lb2gt_6),548).
orientation(lb2gt_4,154,549). coords(lb2gt_4,113,56,549).  visible(lb2gt_4,549). happensAt(walking(lb2gt_4),549). orientation(lb2gt_6,151,549). coords(lb2gt_6,72,122,549).  visible(lb2gt_6,549). happensAt(inactive(lb2gt_6),549).
orientation(lb2gt_4,154,550). coords(lb2gt_4,113,55,550).  visible(lb2gt_4,550). happensAt(walking(lb2gt_4),550). orientation(lb2gt_6,151,550). coords(lb2gt_6,72,122,550).  visible(lb2gt_6,550). happensAt(inactive(lb2gt_6),550).
orientation(lb2gt_4,154,551). coords(lb2gt_4,113,54,551).  visible(lb2gt_4,551). happensAt(walking(lb2gt_4),551). orientation(lb2gt_6,151,551). coords(lb2gt_6,72,122,551).  visible(lb2gt_6,551). happensAt(inactive(lb2gt_6),551).
#show holdsAt/2.