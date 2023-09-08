
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

	orientation(bww1gt_1,173,530). coords(bww1gt_1,100,257,530).  visible(bww1gt_1,530). happensAt(walking(bww1gt_1),530). orientation(bww1gt_2,169,530). coords(bww1gt_2,175,207,530).  visible(bww1gt_2,530). happensAt(walking(bww1gt_2),530).
orientation(bww1gt_1,173,531). coords(bww1gt_1,100,257,531).  visible(bww1gt_1,531). happensAt(walking(bww1gt_1),531). orientation(bww1gt_2,169,531). coords(bww1gt_2,175,207,531).  visible(bww1gt_2,531). happensAt(walking(bww1gt_2),531).
orientation(bww1gt_1,173,532). coords(bww1gt_1,100,257,532).  visible(bww1gt_1,532). happensAt(walking(bww1gt_1),532). orientation(bww1gt_2,169,532). coords(bww1gt_2,174,207,532).  visible(bww1gt_2,532). happensAt(walking(bww1gt_2),532).
orientation(bww1gt_1,173,533). coords(bww1gt_1,100,256,533).  visible(bww1gt_1,533). happensAt(walking(bww1gt_1),533). orientation(bww1gt_2,169,533). coords(bww1gt_2,174,207,533).  visible(bww1gt_2,533). happensAt(walking(bww1gt_2),533).
orientation(bww1gt_1,173,534). coords(bww1gt_1,99,255,534).  visible(bww1gt_1,534). happensAt(walking(bww1gt_1),534). orientation(bww1gt_2,169,534). coords(bww1gt_2,174,208,534).  visible(bww1gt_2,534). happensAt(walking(bww1gt_2),534).
orientation(bww1gt_1,173,535). coords(bww1gt_1,99,255,535).  visible(bww1gt_1,535). happensAt(walking(bww1gt_1),535). orientation(bww1gt_2,169,535). coords(bww1gt_2,173,208,535).  visible(bww1gt_2,535). happensAt(walking(bww1gt_2),535).
orientation(bww1gt_1,173,536). coords(bww1gt_1,99,254,536).  visible(bww1gt_1,536). happensAt(walking(bww1gt_1),536). orientation(bww1gt_2,169,536). coords(bww1gt_2,172,209,536).  visible(bww1gt_2,536). happensAt(walking(bww1gt_2),536).
orientation(bww1gt_1,173,537). coords(bww1gt_1,99,254,537).  visible(bww1gt_1,537). happensAt(walking(bww1gt_1),537). orientation(bww1gt_2,169,537). coords(bww1gt_2,171,210,537).  visible(bww1gt_2,537). happensAt(walking(bww1gt_2),537).
orientation(bww1gt_1,173,538). coords(bww1gt_1,99,254,538).  visible(bww1gt_1,538). happensAt(walking(bww1gt_1),538). orientation(bww1gt_2,169,538). coords(bww1gt_2,171,211,538).  visible(bww1gt_2,538). happensAt(walking(bww1gt_2),538).
orientation(bww1gt_1,173,539). coords(bww1gt_1,99,253,539).  visible(bww1gt_1,539). happensAt(walking(bww1gt_1),539). orientation(bww1gt_2,169,539). coords(bww1gt_2,170,211,539).  visible(bww1gt_2,539). happensAt(walking(bww1gt_2),539).
orientation(bww1gt_1,173,540). coords(bww1gt_1,99,253,540).  visible(bww1gt_1,540). happensAt(walking(bww1gt_1),540). orientation(bww1gt_2,169,540). coords(bww1gt_2,170,212,540).  visible(bww1gt_2,540). happensAt(walking(bww1gt_2),540).
orientation(bww1gt_1,173,541). coords(bww1gt_1,99,253,541).  visible(bww1gt_1,541). happensAt(walking(bww1gt_1),541). orientation(bww1gt_2,169,541). coords(bww1gt_2,169,212,541).  visible(bww1gt_2,541). happensAt(walking(bww1gt_2),541).
orientation(bww1gt_1,173,542). coords(bww1gt_1,99,253,542).  visible(bww1gt_1,542). happensAt(walking(bww1gt_1),542). orientation(bww1gt_2,169,542). coords(bww1gt_2,168,212,542).  visible(bww1gt_2,542). happensAt(walking(bww1gt_2),542).
orientation(bww1gt_1,173,543). coords(bww1gt_1,99,252,543).  visible(bww1gt_1,543). happensAt(walking(bww1gt_1),543). orientation(bww1gt_2,169,543). coords(bww1gt_2,168,212,543).  visible(bww1gt_2,543). happensAt(walking(bww1gt_2),543).
orientation(bww1gt_1,173,544). coords(bww1gt_1,99,252,544).  visible(bww1gt_1,544). happensAt(walking(bww1gt_1),544). orientation(bww1gt_2,169,544). coords(bww1gt_2,167,212,544).  visible(bww1gt_2,544). happensAt(walking(bww1gt_2),544).
orientation(bww1gt_1,173,545). coords(bww1gt_1,99,252,545).  visible(bww1gt_1,545). happensAt(walking(bww1gt_1),545). orientation(bww1gt_2,169,545). coords(bww1gt_2,167,212,545).  visible(bww1gt_2,545). happensAt(walking(bww1gt_2),545).
orientation(bww1gt_1,173,546). coords(bww1gt_1,99,252,546).  visible(bww1gt_1,546). happensAt(walking(bww1gt_1),546). orientation(bww1gt_2,169,546). coords(bww1gt_2,166,211,546).  visible(bww1gt_2,546). happensAt(walking(bww1gt_2),546).
orientation(bww1gt_1,173,547). coords(bww1gt_1,99,252,547).  visible(bww1gt_1,547). happensAt(walking(bww1gt_1),547). orientation(bww1gt_2,169,547). coords(bww1gt_2,166,211,547).  visible(bww1gt_2,547). happensAt(walking(bww1gt_2),547).
orientation(bww1gt_1,173,548). coords(bww1gt_1,99,250,548).  visible(bww1gt_1,548). happensAt(walking(bww1gt_1),548). orientation(bww1gt_2,169,548). coords(bww1gt_2,166,210,548).  visible(bww1gt_2,548). happensAt(walking(bww1gt_2),548).
orientation(bww1gt_1,173,549). coords(bww1gt_1,99,250,549).  visible(bww1gt_1,549). happensAt(walking(bww1gt_1),549). orientation(bww1gt_2,169,549). coords(bww1gt_2,165,209,549).  visible(bww1gt_2,549). happensAt(walking(bww1gt_2),549).
orientation(bww1gt_1,173,550). coords(bww1gt_1,99,250,550).  visible(bww1gt_1,550). happensAt(walking(bww1gt_1),550). orientation(bww1gt_2,169,550). coords(bww1gt_2,165,208,550).  visible(bww1gt_2,550). happensAt(walking(bww1gt_2),550).
orientation(bww1gt_1,173,551). coords(bww1gt_1,99,250,551).  visible(bww1gt_1,551). happensAt(walking(bww1gt_1),551). orientation(bww1gt_2,169,551). coords(bww1gt_2,164,208,551).  visible(bww1gt_2,551). happensAt(walking(bww1gt_2),551).
orientation(bww1gt_1,173,552). coords(bww1gt_1,99,249,552).  visible(bww1gt_1,552). happensAt(inactive(bww1gt_1),552). orientation(bww1gt_2,169,552). coords(bww1gt_2,163,207,552).  visible(bww1gt_2,552). happensAt(walking(bww1gt_2),552).
orientation(bww1gt_1,173,553). coords(bww1gt_1,99,249,553).  visible(bww1gt_1,553). happensAt(inactive(bww1gt_1),553). orientation(bww1gt_2,169,553). coords(bww1gt_2,162,207,553).  visible(bww1gt_2,553). happensAt(walking(bww1gt_2),553).
orientation(bww1gt_1,173,554). coords(bww1gt_1,99,249,554).  visible(bww1gt_1,554). happensAt(inactive(bww1gt_1),554). orientation(bww1gt_2,169,554). coords(bww1gt_2,160,206,554).  visible(bww1gt_2,554). happensAt(walking(bww1gt_2),554).
#show holdsAt/2.