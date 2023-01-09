
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

	orientation(wk1gt_4,157,543). coords(wk1gt_4,56,169,543).  visible(wk1gt_4,543). happensAt(inactive(wk1gt_4),543). orientation(wk1gt_5,162,543). coords(wk1gt_5,57,150,543).  visible(wk1gt_5,543). happensAt(inactive(wk1gt_5),543).
orientation(wk1gt_4,157,544). coords(wk1gt_4,56,169,544).  visible(wk1gt_4,544). happensAt(inactive(wk1gt_4),544). orientation(wk1gt_5,162,544). coords(wk1gt_5,57,150,544).  visible(wk1gt_5,544). happensAt(inactive(wk1gt_5),544).
orientation(wk1gt_4,157,545). coords(wk1gt_4,56,169,545).  visible(wk1gt_4,545). happensAt(inactive(wk1gt_4),545). orientation(wk1gt_5,162,545). coords(wk1gt_5,57,150,545).  visible(wk1gt_5,545). happensAt(inactive(wk1gt_5),545).
orientation(wk1gt_4,157,546). coords(wk1gt_4,56,169,546).  visible(wk1gt_4,546). happensAt(inactive(wk1gt_4),546). orientation(wk1gt_5,162,546). coords(wk1gt_5,57,150,546).  visible(wk1gt_5,546). happensAt(inactive(wk1gt_5),546).
orientation(wk1gt_4,157,547). coords(wk1gt_4,56,169,547).  visible(wk1gt_4,547). happensAt(inactive(wk1gt_4),547). orientation(wk1gt_5,162,547). coords(wk1gt_5,57,150,547).  visible(wk1gt_5,547). happensAt(inactive(wk1gt_5),547).
orientation(wk1gt_4,157,548). coords(wk1gt_4,56,169,548).  visible(wk1gt_4,548). happensAt(inactive(wk1gt_4),548). orientation(wk1gt_5,162,548). coords(wk1gt_5,57,150,548).  visible(wk1gt_5,548). happensAt(inactive(wk1gt_5),548).
orientation(wk1gt_4,157,549). coords(wk1gt_4,56,169,549).  visible(wk1gt_4,549). happensAt(inactive(wk1gt_4),549). orientation(wk1gt_5,162,549). coords(wk1gt_5,57,150,549).  visible(wk1gt_5,549). happensAt(inactive(wk1gt_5),549).
orientation(wk1gt_4,157,550). coords(wk1gt_4,56,169,550).  visible(wk1gt_4,550). happensAt(inactive(wk1gt_4),550). orientation(wk1gt_5,162,550). coords(wk1gt_5,57,150,550).  visible(wk1gt_5,550). happensAt(inactive(wk1gt_5),550).
orientation(wk1gt_4,157,551). coords(wk1gt_4,56,170,551).  visible(wk1gt_4,551). happensAt(inactive(wk1gt_4),551). orientation(wk1gt_5,162,551). coords(wk1gt_5,57,150,551).  visible(wk1gt_5,551). happensAt(inactive(wk1gt_5),551).
orientation(wk1gt_4,157,552). coords(wk1gt_4,56,170,552).  visible(wk1gt_4,552). happensAt(inactive(wk1gt_4),552). orientation(wk1gt_5,162,552). coords(wk1gt_5,57,150,552).  visible(wk1gt_5,552). happensAt(inactive(wk1gt_5),552).
orientation(wk1gt_4,157,553). coords(wk1gt_4,56,170,553).  visible(wk1gt_4,553). happensAt(inactive(wk1gt_4),553). orientation(wk1gt_5,162,553). coords(wk1gt_5,57,150,553).  visible(wk1gt_5,553). happensAt(inactive(wk1gt_5),553).
orientation(wk1gt_4,157,554). coords(wk1gt_4,56,170,554).  visible(wk1gt_4,554). happensAt(inactive(wk1gt_4),554). orientation(wk1gt_5,162,554). coords(wk1gt_5,57,150,554).  visible(wk1gt_5,554). happensAt(inactive(wk1gt_5),554).
orientation(wk1gt_4,157,555). coords(wk1gt_4,56,170,555).  visible(wk1gt_4,555). happensAt(inactive(wk1gt_4),555). orientation(wk1gt_5,162,555). coords(wk1gt_5,57,150,555).  visible(wk1gt_5,555). happensAt(inactive(wk1gt_5),555).
orientation(wk1gt_4,157,556). coords(wk1gt_4,56,170,556).  visible(wk1gt_4,556). happensAt(inactive(wk1gt_4),556). orientation(wk1gt_5,162,556). coords(wk1gt_5,57,150,556).  visible(wk1gt_5,556). happensAt(inactive(wk1gt_5),556).
orientation(wk1gt_4,157,557). coords(wk1gt_4,56,170,557).  visible(wk1gt_4,557). happensAt(inactive(wk1gt_4),557). orientation(wk1gt_5,162,557). coords(wk1gt_5,57,150,557).  visible(wk1gt_5,557). happensAt(inactive(wk1gt_5),557).
orientation(wk1gt_4,157,558). coords(wk1gt_4,56,170,558).  visible(wk1gt_4,558). happensAt(inactive(wk1gt_4),558). orientation(wk1gt_5,162,558). coords(wk1gt_5,57,150,558).  visible(wk1gt_5,558). happensAt(inactive(wk1gt_5),558).
orientation(wk1gt_4,157,559). coords(wk1gt_4,56,170,559).  visible(wk1gt_4,559). happensAt(inactive(wk1gt_4),559). orientation(wk1gt_5,162,559). coords(wk1gt_5,57,150,559).  visible(wk1gt_5,559). happensAt(inactive(wk1gt_5),559).
orientation(wk1gt_4,157,560). coords(wk1gt_4,56,170,560).  visible(wk1gt_4,560). happensAt(inactive(wk1gt_4),560). orientation(wk1gt_5,162,560). coords(wk1gt_5,57,150,560).  visible(wk1gt_5,560). happensAt(inactive(wk1gt_5),560).
orientation(wk1gt_4,157,561). coords(wk1gt_4,56,170,561).  visible(wk1gt_4,561). happensAt(inactive(wk1gt_4),561). orientation(wk1gt_5,162,561). coords(wk1gt_5,57,150,561).  visible(wk1gt_5,561). happensAt(inactive(wk1gt_5),561).
orientation(wk1gt_4,157,562). coords(wk1gt_4,56,170,562).  visible(wk1gt_4,562). happensAt(inactive(wk1gt_4),562). orientation(wk1gt_5,162,562). coords(wk1gt_5,57,150,562).  visible(wk1gt_5,562). happensAt(inactive(wk1gt_5),562).
orientation(wk1gt_4,157,563). coords(wk1gt_4,56,170,563).  visible(wk1gt_4,563). happensAt(inactive(wk1gt_4),563). orientation(wk1gt_5,162,563). coords(wk1gt_5,57,150,563).  visible(wk1gt_5,563). happensAt(inactive(wk1gt_5),563).
orientation(wk1gt_4,157,564). coords(wk1gt_4,56,170,564).  visible(wk1gt_4,564). happensAt(inactive(wk1gt_4),564). orientation(wk1gt_5,162,564). coords(wk1gt_5,57,150,564).  visible(wk1gt_5,564). happensAt(inactive(wk1gt_5),564).
orientation(wk1gt_4,157,565). coords(wk1gt_4,56,170,565).  visible(wk1gt_4,565). happensAt(inactive(wk1gt_4),565). orientation(wk1gt_5,162,565). coords(wk1gt_5,57,150,565).  visible(wk1gt_5,565). happensAt(inactive(wk1gt_5),565).
orientation(wk1gt_4,157,566). coords(wk1gt_4,56,170,566).  visible(wk1gt_4,566). happensAt(inactive(wk1gt_4),566). orientation(wk1gt_5,162,566). coords(wk1gt_5,57,150,566).  visible(wk1gt_5,566). happensAt(inactive(wk1gt_5),566).
orientation(wk1gt_4,157,567). coords(wk1gt_4,56,170,567).  visible(wk1gt_4,567). happensAt(inactive(wk1gt_4),567). orientation(wk1gt_5,162,567). coords(wk1gt_5,57,150,567).  visible(wk1gt_5,567). happensAt(inactive(wk1gt_5),567).
#show holdsAt/2.