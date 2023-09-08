
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

	orientation(fomdgt3_2,80,530). coords(fomdgt3_2,231,99,530).  visible(fomdgt3_2,530). happensAt(walking(fomdgt3_2),530). orientation(fomdgt3_7,159,530). coords(fomdgt3_7,158,203,530).  visible(fomdgt3_7,530). happensAt(walking(fomdgt3_7),530).
orientation(fomdgt3_2,80,531). coords(fomdgt3_2,231,98,531).  visible(fomdgt3_2,531). happensAt(walking(fomdgt3_2),531). orientation(fomdgt3_7,159,531). coords(fomdgt3_7,160,205,531).  visible(fomdgt3_7,531). happensAt(walking(fomdgt3_7),531).
orientation(fomdgt3_2,80,532). coords(fomdgt3_2,231,97,532).  visible(fomdgt3_2,532). happensAt(walking(fomdgt3_2),532). orientation(fomdgt3_7,159,532). coords(fomdgt3_7,163,206,532).  visible(fomdgt3_7,532). happensAt(walking(fomdgt3_7),532).
orientation(fomdgt3_2,80,533). coords(fomdgt3_2,231,97,533).  visible(fomdgt3_2,533). happensAt(walking(fomdgt3_2),533). orientation(fomdgt3_7,159,533). coords(fomdgt3_7,165,207,533).  visible(fomdgt3_7,533). happensAt(walking(fomdgt3_7),533).
orientation(fomdgt3_2,80,534). coords(fomdgt3_2,232,97,534).  visible(fomdgt3_2,534). happensAt(walking(fomdgt3_2),534). orientation(fomdgt3_7,159,534). coords(fomdgt3_7,167,207,534).  visible(fomdgt3_7,534). happensAt(walking(fomdgt3_7),534).
orientation(fomdgt3_2,80,535). coords(fomdgt3_2,232,96,535).  visible(fomdgt3_2,535). happensAt(walking(fomdgt3_2),535). orientation(fomdgt3_7,159,535). coords(fomdgt3_7,168,208,535).  visible(fomdgt3_7,535). happensAt(walking(fomdgt3_7),535).
orientation(fomdgt3_2,80,536). coords(fomdgt3_2,232,96,536).  visible(fomdgt3_2,536). happensAt(walking(fomdgt3_2),536). orientation(fomdgt3_7,159,536). coords(fomdgt3_7,169,209,536).  visible(fomdgt3_7,536). happensAt(walking(fomdgt3_7),536).
orientation(fomdgt3_2,80,537). coords(fomdgt3_2,232,96,537).  visible(fomdgt3_2,537). happensAt(walking(fomdgt3_2),537). orientation(fomdgt3_7,159,537). coords(fomdgt3_7,169,209,537).  visible(fomdgt3_7,537). happensAt(walking(fomdgt3_7),537).
orientation(fomdgt3_2,80,538). coords(fomdgt3_2,232,96,538).  visible(fomdgt3_2,538). happensAt(walking(fomdgt3_2),538). orientation(fomdgt3_7,159,538). coords(fomdgt3_7,169,209,538).  visible(fomdgt3_7,538). happensAt(walking(fomdgt3_7),538).
orientation(fomdgt3_2,80,539). coords(fomdgt3_2,232,96,539).  visible(fomdgt3_2,539). happensAt(walking(fomdgt3_2),539). orientation(fomdgt3_7,159,539). coords(fomdgt3_7,171,210,539).  visible(fomdgt3_7,539). happensAt(walking(fomdgt3_7),539).
orientation(fomdgt3_2,80,540). coords(fomdgt3_2,232,96,540).  visible(fomdgt3_2,540). happensAt(walking(fomdgt3_2),540). orientation(fomdgt3_7,159,540). coords(fomdgt3_7,171,212,540).  visible(fomdgt3_7,540). happensAt(walking(fomdgt3_7),540).
orientation(fomdgt3_2,110,541). coords(fomdgt3_2,235,95,541).  visible(fomdgt3_2,541). happensAt(walking(fomdgt3_2),541). orientation(fomdgt3_7,159,541). coords(fomdgt3_7,171,213,541).  visible(fomdgt3_7,541). happensAt(walking(fomdgt3_7),541).
orientation(fomdgt3_2,110,542). coords(fomdgt3_2,236,94,542).  visible(fomdgt3_2,542). happensAt(walking(fomdgt3_2),542). orientation(fomdgt3_7,159,542). coords(fomdgt3_7,172,213,542).  visible(fomdgt3_7,542). happensAt(walking(fomdgt3_7),542).
orientation(fomdgt3_2,110,543). coords(fomdgt3_2,236,94,543).  visible(fomdgt3_2,543). happensAt(walking(fomdgt3_2),543). orientation(fomdgt3_7,170,543). coords(fomdgt3_7,174,214,543).  visible(fomdgt3_7,543). happensAt(walking(fomdgt3_7),543).
orientation(fomdgt3_2,110,544). coords(fomdgt3_2,238,93,544).  visible(fomdgt3_2,544). happensAt(walking(fomdgt3_2),544). orientation(fomdgt3_7,171,544). coords(fomdgt3_7,177,215,544).  visible(fomdgt3_7,544). happensAt(walking(fomdgt3_7),544).
orientation(fomdgt3_2,110,545). coords(fomdgt3_2,239,92,545).  visible(fomdgt3_2,545). happensAt(walking(fomdgt3_2),545). orientation(fomdgt3_7,171,545). coords(fomdgt3_7,177,216,545).  visible(fomdgt3_7,545). happensAt(walking(fomdgt3_7),545).
orientation(fomdgt3_2,110,546). coords(fomdgt3_2,241,91,546).  visible(fomdgt3_2,546). happensAt(walking(fomdgt3_2),546). orientation(fomdgt3_7,171,546). coords(fomdgt3_7,183,218,546).  visible(fomdgt3_7,546). happensAt(walking(fomdgt3_7),546).
orientation(fomdgt3_2,110,547). coords(fomdgt3_2,243,90,547).  visible(fomdgt3_2,547). happensAt(walking(fomdgt3_2),547). orientation(fomdgt3_7,171,547). coords(fomdgt3_7,185,219,547).  visible(fomdgt3_7,547). happensAt(walking(fomdgt3_7),547).
orientation(fomdgt3_2,110,548). coords(fomdgt3_2,243,90,548).  visible(fomdgt3_2,548). happensAt(walking(fomdgt3_2),548). orientation(fomdgt3_7,171,548). coords(fomdgt3_7,187,220,548).  visible(fomdgt3_7,548). happensAt(walking(fomdgt3_7),548).
orientation(fomdgt3_2,110,549). coords(fomdgt3_2,245,90,549).  visible(fomdgt3_2,549). happensAt(walking(fomdgt3_2),549). orientation(fomdgt3_7,171,549). coords(fomdgt3_7,189,221,549).  visible(fomdgt3_7,549). happensAt(walking(fomdgt3_7),549).
orientation(fomdgt3_2,110,550). coords(fomdgt3_2,245,90,550).  visible(fomdgt3_2,550). happensAt(walking(fomdgt3_2),550). orientation(fomdgt3_7,171,550). coords(fomdgt3_7,191,221,550).  visible(fomdgt3_7,550). happensAt(walking(fomdgt3_7),550).
orientation(fomdgt3_2,110,551). coords(fomdgt3_2,246,91,551).  visible(fomdgt3_2,551). happensAt(walking(fomdgt3_2),551). orientation(fomdgt3_7,171,551). coords(fomdgt3_7,193,221,551).  visible(fomdgt3_7,551). happensAt(walking(fomdgt3_7),551).
orientation(fomdgt3_2,110,552). coords(fomdgt3_2,247,91,552).  visible(fomdgt3_2,552). happensAt(walking(fomdgt3_2),552). orientation(fomdgt3_7,171,552). coords(fomdgt3_7,192,224,552).  visible(fomdgt3_7,552). happensAt(walking(fomdgt3_7),552).
orientation(fomdgt3_2,110,553). coords(fomdgt3_2,248,91,553).  visible(fomdgt3_2,553). happensAt(walking(fomdgt3_2),553). orientation(fomdgt3_7,171,553). coords(fomdgt3_7,194,224,553).  visible(fomdgt3_7,553). happensAt(walking(fomdgt3_7),553).
orientation(fomdgt3_2,110,554). coords(fomdgt3_2,249,91,554).  visible(fomdgt3_2,554). happensAt(walking(fomdgt3_2),554). orientation(fomdgt3_7,171,554). coords(fomdgt3_7,195,224,554).  visible(fomdgt3_7,554). happensAt(walking(fomdgt3_7),554).
#show holdsAt/2.