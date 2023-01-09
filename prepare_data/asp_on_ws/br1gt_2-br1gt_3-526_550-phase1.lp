
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

	orientation(br1gt_2,12,526). coords(br1gt_2,86,209,526).  visible(br1gt_2,526). happensAt(inactive(br1gt_2),526). orientation(br1gt_3,155,526). coords(br1gt_3,142,123,526).  visible(br1gt_3,526). happensAt(walking(br1gt_3),526).
orientation(br1gt_2,12,527). coords(br1gt_2,86,209,527).  visible(br1gt_2,527). happensAt(inactive(br1gt_2),527). orientation(br1gt_3,155,527). coords(br1gt_3,145,123,527).  visible(br1gt_3,527). happensAt(walking(br1gt_3),527).
orientation(br1gt_2,12,528). coords(br1gt_2,86,209,528).  visible(br1gt_2,528). happensAt(inactive(br1gt_2),528). orientation(br1gt_3,155,528). coords(br1gt_3,146,122,528).  visible(br1gt_3,528). happensAt(walking(br1gt_3),528).
orientation(br1gt_2,12,529). coords(br1gt_2,86,209,529).  visible(br1gt_2,529). happensAt(inactive(br1gt_2),529). orientation(br1gt_3,155,529). coords(br1gt_3,149,122,529).  visible(br1gt_3,529). happensAt(walking(br1gt_3),529).
orientation(br1gt_2,12,530). coords(br1gt_2,86,209,530).  visible(br1gt_2,530). happensAt(inactive(br1gt_2),530). orientation(br1gt_3,155,530). coords(br1gt_3,150,122,530).  visible(br1gt_3,530). happensAt(walking(br1gt_3),530).
orientation(br1gt_2,12,531). coords(br1gt_2,86,209,531).  visible(br1gt_2,531). happensAt(inactive(br1gt_2),531). orientation(br1gt_3,155,531). coords(br1gt_3,152,121,531).  visible(br1gt_3,531). happensAt(walking(br1gt_3),531).
orientation(br1gt_2,12,532). coords(br1gt_2,86,209,532).  visible(br1gt_2,532). happensAt(inactive(br1gt_2),532). orientation(br1gt_3,155,532). coords(br1gt_3,152,121,532).  visible(br1gt_3,532). happensAt(walking(br1gt_3),532).
orientation(br1gt_2,12,533). coords(br1gt_2,86,209,533).  visible(br1gt_2,533). happensAt(inactive(br1gt_2),533). orientation(br1gt_3,152,533). coords(br1gt_3,153,121,533).  visible(br1gt_3,533). happensAt(walking(br1gt_3),533).
orientation(br1gt_2,12,534). coords(br1gt_2,86,209,534).  visible(br1gt_2,534). happensAt(inactive(br1gt_2),534). orientation(br1gt_3,152,534). coords(br1gt_3,154,120,534).  visible(br1gt_3,534). happensAt(walking(br1gt_3),534).
orientation(br1gt_2,12,535). coords(br1gt_2,86,209,535).  visible(br1gt_2,535). happensAt(inactive(br1gt_2),535). orientation(br1gt_3,152,535). coords(br1gt_3,154,120,535).  visible(br1gt_3,535). happensAt(walking(br1gt_3),535).
orientation(br1gt_2,12,536). coords(br1gt_2,86,209,536).  visible(br1gt_2,536). happensAt(inactive(br1gt_2),536). orientation(br1gt_3,152,536). coords(br1gt_3,155,119,536).  visible(br1gt_3,536). happensAt(walking(br1gt_3),536).
orientation(br1gt_2,12,537). coords(br1gt_2,86,209,537).  visible(br1gt_2,537). happensAt(inactive(br1gt_2),537). orientation(br1gt_3,152,537). coords(br1gt_3,156,119,537).  visible(br1gt_3,537). happensAt(walking(br1gt_3),537).
orientation(br1gt_2,12,538). coords(br1gt_2,86,209,538).  visible(br1gt_2,538). happensAt(inactive(br1gt_2),538). orientation(br1gt_3,152,538). coords(br1gt_3,160,119,538).  visible(br1gt_3,538). happensAt(walking(br1gt_3),538).
orientation(br1gt_2,12,539). coords(br1gt_2,86,209,539).  visible(br1gt_2,539). happensAt(inactive(br1gt_2),539). orientation(br1gt_3,152,539). coords(br1gt_3,163,119,539).  visible(br1gt_3,539). happensAt(walking(br1gt_3),539).
orientation(br1gt_2,12,540). coords(br1gt_2,86,209,540).  visible(br1gt_2,540). happensAt(inactive(br1gt_2),540). orientation(br1gt_3,152,540). coords(br1gt_3,165,119,540).  visible(br1gt_3,540). happensAt(walking(br1gt_3),540).
orientation(br1gt_2,12,541). coords(br1gt_2,86,209,541).  visible(br1gt_2,541). happensAt(inactive(br1gt_2),541). orientation(br1gt_3,152,541). coords(br1gt_3,168,118,541).  visible(br1gt_3,541). happensAt(walking(br1gt_3),541).
orientation(br1gt_2,12,542). coords(br1gt_2,86,209,542).  visible(br1gt_2,542). happensAt(inactive(br1gt_2),542). orientation(br1gt_3,152,542). coords(br1gt_3,170,118,542).  visible(br1gt_3,542). happensAt(walking(br1gt_3),542).
orientation(br1gt_2,12,543). coords(br1gt_2,86,209,543).  visible(br1gt_2,543). happensAt(inactive(br1gt_2),543). orientation(br1gt_3,154,543). coords(br1gt_3,172,117,543).  visible(br1gt_3,543). happensAt(walking(br1gt_3),543).
orientation(br1gt_2,12,544). coords(br1gt_2,86,209,544).  visible(br1gt_2,544). happensAt(inactive(br1gt_2),544). orientation(br1gt_3,154,544). coords(br1gt_3,173,117,544).  visible(br1gt_3,544). happensAt(walking(br1gt_3),544).
orientation(br1gt_2,12,545). coords(br1gt_2,86,209,545).  visible(br1gt_2,545). happensAt(inactive(br1gt_2),545). orientation(br1gt_3,150,545). coords(br1gt_3,174,116,545).  visible(br1gt_3,545). happensAt(walking(br1gt_3),545).
orientation(br1gt_2,12,546). coords(br1gt_2,86,209,546).  visible(br1gt_2,546). happensAt(inactive(br1gt_2),546). orientation(br1gt_3,150,546). coords(br1gt_3,175,115,546).  visible(br1gt_3,546). happensAt(walking(br1gt_3),546).
orientation(br1gt_2,12,547). coords(br1gt_2,86,209,547).  visible(br1gt_2,547). happensAt(inactive(br1gt_2),547). orientation(br1gt_3,150,547). coords(br1gt_3,176,114,547).  visible(br1gt_3,547). happensAt(walking(br1gt_3),547).
orientation(br1gt_2,12,548). coords(br1gt_2,86,209,548).  visible(br1gt_2,548). happensAt(inactive(br1gt_2),548). orientation(br1gt_3,150,548). coords(br1gt_3,177,114,548).  visible(br1gt_3,548). happensAt(walking(br1gt_3),548).
orientation(br1gt_2,12,549). coords(br1gt_2,86,209,549).  visible(br1gt_2,549). happensAt(inactive(br1gt_2),549). orientation(br1gt_3,145,549). coords(br1gt_3,177,113,549).  visible(br1gt_3,549). happensAt(walking(br1gt_3),549).
orientation(br1gt_2,12,550). coords(br1gt_2,86,209,550).  visible(br1gt_2,550). happensAt(inactive(br1gt_2),550). orientation(br1gt_3,142,550). coords(br1gt_3,178,112,550).  visible(br1gt_3,550). happensAt(walking(br1gt_3),550).
#show holdsAt/2.