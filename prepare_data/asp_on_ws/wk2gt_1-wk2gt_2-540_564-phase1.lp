
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
orientation(wk2gt_1,122,553). coords(wk2gt_1,115,53,553).  visible(wk2gt_1,553). happensAt(inactive(wk2gt_1),553). orientation(wk2gt_2,142,553). coords(wk2gt_2,99,63,553).  visible(wk2gt_2,553). happensAt(inactive(wk2gt_2),553).
orientation(wk2gt_1,122,554). coords(wk2gt_1,115,53,554).  visible(wk2gt_1,554). happensAt(inactive(wk2gt_1),554). orientation(wk2gt_2,142,554). coords(wk2gt_2,99,63,554).  visible(wk2gt_2,554). happensAt(inactive(wk2gt_2),554).
orientation(wk2gt_1,122,555). coords(wk2gt_1,115,53,555).  visible(wk2gt_1,555). happensAt(inactive(wk2gt_1),555). orientation(wk2gt_2,142,555). coords(wk2gt_2,99,63,555).  visible(wk2gt_2,555). happensAt(inactive(wk2gt_2),555).
orientation(wk2gt_1,122,556). coords(wk2gt_1,115,53,556).  visible(wk2gt_1,556). happensAt(inactive(wk2gt_1),556). orientation(wk2gt_2,142,556). coords(wk2gt_2,99,63,556).  visible(wk2gt_2,556). happensAt(inactive(wk2gt_2),556).
orientation(wk2gt_1,122,557). coords(wk2gt_1,115,53,557).  visible(wk2gt_1,557). happensAt(inactive(wk2gt_1),557). orientation(wk2gt_2,142,557). coords(wk2gt_2,99,63,557).  visible(wk2gt_2,557). happensAt(inactive(wk2gt_2),557).
orientation(wk2gt_1,122,558). coords(wk2gt_1,115,53,558).  visible(wk2gt_1,558). happensAt(inactive(wk2gt_1),558). orientation(wk2gt_2,142,558). coords(wk2gt_2,99,63,558).  visible(wk2gt_2,558). happensAt(inactive(wk2gt_2),558).
orientation(wk2gt_1,122,559). coords(wk2gt_1,115,53,559).  visible(wk2gt_1,559). happensAt(inactive(wk2gt_1),559). orientation(wk2gt_2,142,559). coords(wk2gt_2,99,63,559).  visible(wk2gt_2,559). happensAt(inactive(wk2gt_2),559).
orientation(wk2gt_1,122,560). coords(wk2gt_1,115,53,560).  visible(wk2gt_1,560). happensAt(inactive(wk2gt_1),560). orientation(wk2gt_2,142,560). coords(wk2gt_2,99,63,560).  visible(wk2gt_2,560). happensAt(inactive(wk2gt_2),560).
orientation(wk2gt_1,125,561). coords(wk2gt_1,115,53,561).  visible(wk2gt_1,561). happensAt(inactive(wk2gt_1),561). orientation(wk2gt_2,142,561). coords(wk2gt_2,99,63,561).  visible(wk2gt_2,561). happensAt(inactive(wk2gt_2),561).
orientation(wk2gt_1,125,562). coords(wk2gt_1,115,53,562).  visible(wk2gt_1,562). happensAt(inactive(wk2gt_1),562). orientation(wk2gt_2,142,562). coords(wk2gt_2,99,63,562).  visible(wk2gt_2,562). happensAt(inactive(wk2gt_2),562).
orientation(wk2gt_1,125,563). coords(wk2gt_1,115,53,563).  visible(wk2gt_1,563). happensAt(inactive(wk2gt_1),563). orientation(wk2gt_2,142,563). coords(wk2gt_2,99,63,563).  visible(wk2gt_2,563). happensAt(inactive(wk2gt_2),563).
orientation(wk2gt_1,125,564). coords(wk2gt_1,115,53,564).  visible(wk2gt_1,564). happensAt(inactive(wk2gt_1),564). orientation(wk2gt_2,142,564). coords(wk2gt_2,99,63,564).  visible(wk2gt_2,564). happensAt(inactive(wk2gt_2),564).
#show holdsAt/2.