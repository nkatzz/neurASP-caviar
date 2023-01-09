
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

	orientation(mwt2gt_1,153,541). coords(mwt2gt_1,42,155,541).  visible(mwt2gt_1,541). happensAt(walking(mwt2gt_1),541). orientation(mwt2gt_2,152,541). coords(mwt2gt_2,36,168,541).  visible(mwt2gt_2,541). happensAt(walking(mwt2gt_2),541).
orientation(mwt2gt_1,153,542). coords(mwt2gt_1,42,155,542).  visible(mwt2gt_1,542). happensAt(walking(mwt2gt_1),542). orientation(mwt2gt_2,152,542). coords(mwt2gt_2,35,168,542).  visible(mwt2gt_2,542). happensAt(walking(mwt2gt_2),542).
orientation(mwt2gt_1,153,543). coords(mwt2gt_1,41,155,543).  visible(mwt2gt_1,543). happensAt(walking(mwt2gt_1),543). orientation(mwt2gt_2,152,543). coords(mwt2gt_2,35,168,543).  visible(mwt2gt_2,543). happensAt(walking(mwt2gt_2),543).
orientation(mwt2gt_1,153,544). coords(mwt2gt_1,40,155,544).  visible(mwt2gt_1,544). happensAt(walking(mwt2gt_1),544). orientation(mwt2gt_2,152,544). coords(mwt2gt_2,35,168,544).  visible(mwt2gt_2,544). happensAt(walking(mwt2gt_2),544).
orientation(mwt2gt_1,153,545). coords(mwt2gt_1,40,155,545).  visible(mwt2gt_1,545). happensAt(walking(mwt2gt_1),545). orientation(mwt2gt_2,152,545). coords(mwt2gt_2,35,168,545).  visible(mwt2gt_2,545). happensAt(walking(mwt2gt_2),545).
orientation(mwt2gt_1,153,546). coords(mwt2gt_1,39,155,546).  visible(mwt2gt_1,546). happensAt(walking(mwt2gt_1),546). orientation(mwt2gt_2,152,546). coords(mwt2gt_2,35,167,546).  visible(mwt2gt_2,546). happensAt(walking(mwt2gt_2),546).
orientation(mwt2gt_1,153,547). coords(mwt2gt_1,39,155,547).  visible(mwt2gt_1,547). happensAt(walking(mwt2gt_1),547). orientation(mwt2gt_2,152,547). coords(mwt2gt_2,35,167,547).  visible(mwt2gt_2,547). happensAt(walking(mwt2gt_2),547).
orientation(mwt2gt_1,153,548). coords(mwt2gt_1,39,155,548).  visible(mwt2gt_1,548). happensAt(walking(mwt2gt_1),548). orientation(mwt2gt_2,152,548). coords(mwt2gt_2,35,167,548).  visible(mwt2gt_2,548). happensAt(walking(mwt2gt_2),548).
orientation(mwt2gt_1,153,549). coords(mwt2gt_1,39,155,549).  visible(mwt2gt_1,549). happensAt(walking(mwt2gt_1),549). orientation(mwt2gt_2,152,549). coords(mwt2gt_2,34,167,549).  visible(mwt2gt_2,549). happensAt(walking(mwt2gt_2),549).
orientation(mwt2gt_1,153,550). coords(mwt2gt_1,38,155,550).  visible(mwt2gt_1,550). happensAt(walking(mwt2gt_1),550). orientation(mwt2gt_2,152,550). coords(mwt2gt_2,34,167,550).  visible(mwt2gt_2,550). happensAt(walking(mwt2gt_2),550).
orientation(mwt2gt_1,153,551). coords(mwt2gt_1,38,155,551).  visible(mwt2gt_1,551). happensAt(walking(mwt2gt_1),551). orientation(mwt2gt_2,152,551). coords(mwt2gt_2,34,167,551).  visible(mwt2gt_2,551). happensAt(walking(mwt2gt_2),551).
orientation(mwt2gt_1,153,552). coords(mwt2gt_1,38,155,552).  visible(mwt2gt_1,552). happensAt(walking(mwt2gt_1),552). orientation(mwt2gt_2,152,552). coords(mwt2gt_2,34,167,552).  visible(mwt2gt_2,552). happensAt(walking(mwt2gt_2),552).
orientation(mwt2gt_1,153,553). coords(mwt2gt_1,38,155,553).  visible(mwt2gt_1,553). happensAt(walking(mwt2gt_1),553). orientation(mwt2gt_2,152,553). coords(mwt2gt_2,33,167,553).  visible(mwt2gt_2,553). happensAt(walking(mwt2gt_2),553).
orientation(mwt2gt_1,153,554). coords(mwt2gt_1,38,155,554).  visible(mwt2gt_1,554). happensAt(walking(mwt2gt_1),554). orientation(mwt2gt_2,152,554). coords(mwt2gt_2,33,167,554).  visible(mwt2gt_2,554). happensAt(walking(mwt2gt_2),554).
orientation(mwt2gt_1,153,555). coords(mwt2gt_1,38,155,555).  visible(mwt2gt_1,555). happensAt(walking(mwt2gt_1),555). orientation(mwt2gt_2,152,555). coords(mwt2gt_2,32,167,555).  visible(mwt2gt_2,555). happensAt(walking(mwt2gt_2),555).
orientation(mwt2gt_1,153,556). coords(mwt2gt_1,38,155,556).  visible(mwt2gt_1,556). happensAt(walking(mwt2gt_1),556). orientation(mwt2gt_2,152,556). coords(mwt2gt_2,32,167,556).  visible(mwt2gt_2,556). happensAt(walking(mwt2gt_2),556).
orientation(mwt2gt_1,153,557). coords(mwt2gt_1,37,155,557).  visible(mwt2gt_1,557). happensAt(walking(mwt2gt_1),557). orientation(mwt2gt_2,152,557). coords(mwt2gt_2,32,167,557).  visible(mwt2gt_2,557). happensAt(walking(mwt2gt_2),557).
orientation(mwt2gt_1,153,558). coords(mwt2gt_1,36,155,558).  visible(mwt2gt_1,558). happensAt(walking(mwt2gt_1),558). orientation(mwt2gt_2,152,558). coords(mwt2gt_2,32,167,558).  visible(mwt2gt_2,558). happensAt(walking(mwt2gt_2),558).
orientation(mwt2gt_1,153,559). coords(mwt2gt_1,36,155,559).  visible(mwt2gt_1,559). happensAt(walking(mwt2gt_1),559). orientation(mwt2gt_2,152,559). coords(mwt2gt_2,32,167,559).  visible(mwt2gt_2,559). happensAt(walking(mwt2gt_2),559).
orientation(mwt2gt_1,153,560). coords(mwt2gt_1,36,155,560).  visible(mwt2gt_1,560). happensAt(walking(mwt2gt_1),560). orientation(mwt2gt_2,152,560). coords(mwt2gt_2,32,167,560).  visible(mwt2gt_2,560). happensAt(walking(mwt2gt_2),560).
orientation(mwt2gt_1,153,561). coords(mwt2gt_1,36,155,561).  visible(mwt2gt_1,561). happensAt(walking(mwt2gt_1),561). orientation(mwt2gt_2,152,561). coords(mwt2gt_2,32,167,561).  visible(mwt2gt_2,561). happensAt(walking(mwt2gt_2),561).
orientation(mwt2gt_1,153,562). coords(mwt2gt_1,36,155,562).  visible(mwt2gt_1,562). happensAt(walking(mwt2gt_1),562). orientation(mwt2gt_2,152,562). coords(mwt2gt_2,32,167,562).  visible(mwt2gt_2,562). happensAt(walking(mwt2gt_2),562).
orientation(mwt2gt_1,153,563). coords(mwt2gt_1,36,155,563).  visible(mwt2gt_1,563). happensAt(walking(mwt2gt_1),563). orientation(mwt2gt_2,152,563). coords(mwt2gt_2,32,167,563).  visible(mwt2gt_2,563). happensAt(walking(mwt2gt_2),563).
orientation(mwt2gt_1,153,564). coords(mwt2gt_1,36,155,564).  visible(mwt2gt_1,564). happensAt(walking(mwt2gt_1),564). orientation(mwt2gt_2,152,564). coords(mwt2gt_2,31,167,564).  visible(mwt2gt_2,564). happensAt(walking(mwt2gt_2),564).
orientation(mwt2gt_1,153,565). coords(mwt2gt_1,36,155,565).  visible(mwt2gt_1,565). happensAt(walking(mwt2gt_1),565). orientation(mwt2gt_2,152,565). coords(mwt2gt_2,31,167,565).  visible(mwt2gt_2,565). happensAt(walking(mwt2gt_2),565).
#show holdsAt/2.