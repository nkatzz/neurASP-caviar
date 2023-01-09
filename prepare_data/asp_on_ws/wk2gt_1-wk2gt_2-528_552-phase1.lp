
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

	orientation(wk2gt_1,122,528). coords(wk2gt_1,115,53,528).  visible(wk2gt_1,528). happensAt(inactive(wk2gt_1),528). orientation(wk2gt_2,141,528). coords(wk2gt_2,99,63,528).  visible(wk2gt_2,528). happensAt(inactive(wk2gt_2),528).
orientation(wk2gt_1,122,529). coords(wk2gt_1,115,53,529).  visible(wk2gt_1,529). happensAt(inactive(wk2gt_1),529). orientation(wk2gt_2,141,529). coords(wk2gt_2,99,63,529).  visible(wk2gt_2,529). happensAt(inactive(wk2gt_2),529).
orientation(wk2gt_1,122,530). coords(wk2gt_1,115,53,530).  visible(wk2gt_1,530). happensAt(inactive(wk2gt_1),530). orientation(wk2gt_2,141,530). coords(wk2gt_2,99,63,530).  visible(wk2gt_2,530). happensAt(inactive(wk2gt_2),530).
orientation(wk2gt_1,122,531). coords(wk2gt_1,115,53,531).  visible(wk2gt_1,531). happensAt(inactive(wk2gt_1),531). orientation(wk2gt_2,141,531). coords(wk2gt_2,99,63,531).  visible(wk2gt_2,531). happensAt(inactive(wk2gt_2),531).
orientation(wk2gt_1,122,532). coords(wk2gt_1,115,53,532).  visible(wk2gt_1,532). happensAt(inactive(wk2gt_1),532). orientation(wk2gt_2,141,532). coords(wk2gt_2,99,63,532).  visible(wk2gt_2,532). happensAt(inactive(wk2gt_2),532).
orientation(wk2gt_1,122,533). coords(wk2gt_1,115,53,533).  visible(wk2gt_1,533). happensAt(inactive(wk2gt_1),533). orientation(wk2gt_2,141,533). coords(wk2gt_2,99,63,533).  visible(wk2gt_2,533). happensAt(inactive(wk2gt_2),533).
orientation(wk2gt_1,122,534). coords(wk2gt_1,115,53,534).  visible(wk2gt_1,534). happensAt(inactive(wk2gt_1),534). orientation(wk2gt_2,141,534). coords(wk2gt_2,99,63,534).  visible(wk2gt_2,534). happensAt(inactive(wk2gt_2),534).
orientation(wk2gt_1,122,535). coords(wk2gt_1,115,53,535).  visible(wk2gt_1,535). happensAt(inactive(wk2gt_1),535). orientation(wk2gt_2,141,535). coords(wk2gt_2,99,63,535).  visible(wk2gt_2,535). happensAt(inactive(wk2gt_2),535).
orientation(wk2gt_1,122,536). coords(wk2gt_1,115,53,536).  visible(wk2gt_1,536). happensAt(inactive(wk2gt_1),536). orientation(wk2gt_2,141,536). coords(wk2gt_2,99,63,536).  visible(wk2gt_2,536). happensAt(inactive(wk2gt_2),536).
orientation(wk2gt_1,122,537). coords(wk2gt_1,115,53,537).  visible(wk2gt_1,537). happensAt(inactive(wk2gt_1),537). orientation(wk2gt_2,141,537). coords(wk2gt_2,99,63,537).  visible(wk2gt_2,537). happensAt(inactive(wk2gt_2),537).
orientation(wk2gt_1,122,538). coords(wk2gt_1,115,53,538).  visible(wk2gt_1,538). happensAt(inactive(wk2gt_1),538). orientation(wk2gt_2,141,538). coords(wk2gt_2,99,63,538).  visible(wk2gt_2,538). happensAt(inactive(wk2gt_2),538).
orientation(wk2gt_1,122,539). coords(wk2gt_1,115,53,539).  visible(wk2gt_1,539). happensAt(inactive(wk2gt_1),539). orientation(wk2gt_2,141,539). coords(wk2gt_2,99,63,539).  visible(wk2gt_2,539). happensAt(inactive(wk2gt_2),539).
orientation(wk2gt_1,122,540). coords(wk2gt_1,115,53,540).  visible(wk2gt_1,540). happensAt(inactive(wk2gt_1),540). orientation(wk2gt_2,141,540). coords(wk2gt_2,99,63,540).  visible(wk2gt_2,540). happensAt(inactive(wk2gt_2),540).
orientation(wk2gt_1,122,541). coords(wk2gt_1,115,53,541).  visible(wk2gt_1,541). happensAt(inactive(wk2gt_1),541). orientation(wk2gt_2,142,541). coords(wk2gt_2,99,63,541).  visible(wk2gt_2,541). happensAt(inactive(wk2gt_2),541).
orientation(wk2gt_1,122,542). coords(wk2gt_1,115,53,542).  visible(wk2gt_1,542). happensAt(inactive(wk2gt_1),542). orientation(wk2gt_2,142,542). coords(wk2gt_2,99,63,542).  visible(wk2gt_2,542). happensAt(inactive(wk2gt_2),542).
orientation(wk2gt_1,122,543). coords(wk2gt_1,115,53,543).  visible(wk2gt_1,543). happensAt(inactive(wk2gt_1),543). orientation(wk2gt_2,142,543). coords(wk2gt_2,99,63,543).  visible(wk2gt_2,543). happensAt(inactive(wk2gt_2),543).
orientation(wk2gt_1,122,544). coords(wk2gt_1,115,53,544).  visible(wk2gt_1,544). happensAt(inactive(wk2gt_1),544). orientation(wk2gt_2,142,544). coords(wk2gt_2,99,63,544).  visible(wk2gt_2,544). happensAt(inactive(wk2gt_2),544).
orientation(wk2gt_1,122,545). coords(wk2gt_1,115,53,545).  visible(wk2gt_1,545). happensAt(inactive(wk2gt_1),545). orientation(wk2gt_2,142,545). coords(wk2gt_2,99,63,545).  visible(wk2gt_2,545). happensAt(inactive(wk2gt_2),545).
orientation(wk2gt_1,122,546). coords(wk2gt_1,115,53,546).  visible(wk2gt_1,546). happensAt(inactive(wk2gt_1),546). orientation(wk2gt_2,142,546). coords(wk2gt_2,99,63,546).  visible(wk2gt_2,546). happensAt(inactive(wk2gt_2),546).
orientation(wk2gt_1,122,547). coords(wk2gt_1,115,53,547).  visible(wk2gt_1,547). happensAt(inactive(wk2gt_1),547). orientation(wk2gt_2,142,547). coords(wk2gt_2,99,63,547).  visible(wk2gt_2,547). happensAt(inactive(wk2gt_2),547).
orientation(wk2gt_1,122,548). coords(wk2gt_1,115,53,548).  visible(wk2gt_1,548). happensAt(inactive(wk2gt_1),548). orientation(wk2gt_2,142,548). coords(wk2gt_2,99,63,548).  visible(wk2gt_2,548). happensAt(inactive(wk2gt_2),548).
orientation(wk2gt_1,122,549). coords(wk2gt_1,115,53,549).  visible(wk2gt_1,549). happensAt(inactive(wk2gt_1),549). orientation(wk2gt_2,142,549). coords(wk2gt_2,99,63,549).  visible(wk2gt_2,549). happensAt(inactive(wk2gt_2),549).
orientation(wk2gt_1,122,550). coords(wk2gt_1,115,53,550).  visible(wk2gt_1,550). happensAt(inactive(wk2gt_1),550). orientation(wk2gt_2,142,550). coords(wk2gt_2,99,63,550).  visible(wk2gt_2,550). happensAt(inactive(wk2gt_2),550).
orientation(wk2gt_1,122,551). coords(wk2gt_1,115,53,551).  visible(wk2gt_1,551). happensAt(inactive(wk2gt_1),551). orientation(wk2gt_2,142,551). coords(wk2gt_2,99,63,551).  visible(wk2gt_2,551). happensAt(inactive(wk2gt_2),551).
orientation(wk2gt_1,122,552). coords(wk2gt_1,115,53,552).  visible(wk2gt_1,552). happensAt(inactive(wk2gt_1),552). orientation(wk2gt_2,142,552). coords(wk2gt_2,99,63,552).  visible(wk2gt_2,552). happensAt(inactive(wk2gt_2),552).
#show holdsAt/2.