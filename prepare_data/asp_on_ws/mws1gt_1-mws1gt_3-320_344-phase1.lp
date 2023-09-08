
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

	orientation(mws1gt_1,155,320). coords(mws1gt_1,76,100,320).  visible(mws1gt_1,320). happensAt(walking(mws1gt_1),320). orientation(mws1gt_3,158,320). coords(mws1gt_3,64,140,320).  visible(mws1gt_3,320). happensAt(walking(mws1gt_3),320).
orientation(mws1gt_1,155,321). coords(mws1gt_1,76,100,321).  visible(mws1gt_1,321). happensAt(walking(mws1gt_1),321). orientation(mws1gt_3,158,321). coords(mws1gt_3,64,140,321).  visible(mws1gt_3,321). happensAt(walking(mws1gt_3),321).
orientation(mws1gt_1,155,322). coords(mws1gt_1,76,100,322).  visible(mws1gt_1,322). happensAt(walking(mws1gt_1),322). orientation(mws1gt_3,165,322). coords(mws1gt_3,64,140,322).  visible(mws1gt_3,322). happensAt(walking(mws1gt_3),322).
orientation(mws1gt_1,155,323). coords(mws1gt_1,76,100,323).  visible(mws1gt_1,323). happensAt(walking(mws1gt_1),323). orientation(mws1gt_3,165,323). coords(mws1gt_3,63,141,323).  visible(mws1gt_3,323). happensAt(walking(mws1gt_3),323).
orientation(mws1gt_1,155,324). coords(mws1gt_1,75,100,324).  visible(mws1gt_1,324). happensAt(walking(mws1gt_1),324). orientation(mws1gt_3,165,324). coords(mws1gt_3,63,141,324).  visible(mws1gt_3,324). happensAt(walking(mws1gt_3),324).
orientation(mws1gt_1,155,325). coords(mws1gt_1,75,100,325).  visible(mws1gt_1,325). happensAt(walking(mws1gt_1),325). orientation(mws1gt_3,165,325). coords(mws1gt_3,63,141,325).  visible(mws1gt_3,325). happensAt(walking(mws1gt_3),325).
orientation(mws1gt_1,155,326). coords(mws1gt_1,75,100,326).  visible(mws1gt_1,326). happensAt(walking(mws1gt_1),326). orientation(mws1gt_3,165,326). coords(mws1gt_3,63,142,326).  visible(mws1gt_3,326). happensAt(walking(mws1gt_3),326).
orientation(mws1gt_1,155,327). coords(mws1gt_1,75,100,327).  visible(mws1gt_1,327). happensAt(walking(mws1gt_1),327). orientation(mws1gt_3,165,327). coords(mws1gt_3,62,143,327).  visible(mws1gt_3,327). happensAt(walking(mws1gt_3),327).
orientation(mws1gt_1,155,328). coords(mws1gt_1,74,100,328).  visible(mws1gt_1,328). happensAt(walking(mws1gt_1),328). orientation(mws1gt_3,165,328). coords(mws1gt_3,61,143,328).  visible(mws1gt_3,328). happensAt(walking(mws1gt_3),328).
orientation(mws1gt_1,155,329). coords(mws1gt_1,74,99,329).  visible(mws1gt_1,329). happensAt(walking(mws1gt_1),329). orientation(mws1gt_3,165,329). coords(mws1gt_3,61,143,329).  visible(mws1gt_3,329). happensAt(walking(mws1gt_3),329).
orientation(mws1gt_1,155,330). coords(mws1gt_1,73,99,330).  visible(mws1gt_1,330). happensAt(walking(mws1gt_1),330). orientation(mws1gt_3,165,330). coords(mws1gt_3,60,144,330).  visible(mws1gt_3,330). happensAt(walking(mws1gt_3),330).
orientation(mws1gt_1,155,331). coords(mws1gt_1,72,99,331).  visible(mws1gt_1,331). happensAt(walking(mws1gt_1),331). orientation(mws1gt_3,165,331). coords(mws1gt_3,60,146,331).  visible(mws1gt_3,331). happensAt(walking(mws1gt_3),331).
orientation(mws1gt_1,155,332). coords(mws1gt_1,71,99,332).  visible(mws1gt_1,332). happensAt(walking(mws1gt_1),332). orientation(mws1gt_3,165,332). coords(mws1gt_3,60,146,332).  visible(mws1gt_3,332). happensAt(walking(mws1gt_3),332).
orientation(mws1gt_1,155,333). coords(mws1gt_1,71,99,333).  visible(mws1gt_1,333). happensAt(walking(mws1gt_1),333). orientation(mws1gt_3,165,333). coords(mws1gt_3,60,147,333).  visible(mws1gt_3,333). happensAt(walking(mws1gt_3),333).
orientation(mws1gt_1,155,334). coords(mws1gt_1,71,99,334).  visible(mws1gt_1,334). happensAt(walking(mws1gt_1),334). orientation(mws1gt_3,165,334). coords(mws1gt_3,60,147,334).  visible(mws1gt_3,334). happensAt(walking(mws1gt_3),334).
orientation(mws1gt_1,155,335). coords(mws1gt_1,71,99,335).  visible(mws1gt_1,335). happensAt(walking(mws1gt_1),335). orientation(mws1gt_3,165,335). coords(mws1gt_3,60,148,335).  visible(mws1gt_3,335). happensAt(walking(mws1gt_3),335).
orientation(mws1gt_1,155,336). coords(mws1gt_1,71,98,336).  visible(mws1gt_1,336). happensAt(walking(mws1gt_1),336). orientation(mws1gt_3,165,336). coords(mws1gt_3,60,149,336).  visible(mws1gt_3,336). happensAt(walking(mws1gt_3),336).
orientation(mws1gt_1,155,337). coords(mws1gt_1,71,98,337).  visible(mws1gt_1,337). happensAt(walking(mws1gt_1),337). orientation(mws1gt_3,165,337). coords(mws1gt_3,59,149,337).  visible(mws1gt_3,337). happensAt(walking(mws1gt_3),337).
orientation(mws1gt_1,155,338). coords(mws1gt_1,71,98,338).  visible(mws1gt_1,338). happensAt(walking(mws1gt_1),338). orientation(mws1gt_3,165,338). coords(mws1gt_3,59,149,338).  visible(mws1gt_3,338). happensAt(walking(mws1gt_3),338).
orientation(mws1gt_1,155,339). coords(mws1gt_1,71,97,339).  visible(mws1gt_1,339). happensAt(walking(mws1gt_1),339). orientation(mws1gt_3,165,339). coords(mws1gt_3,59,149,339).  visible(mws1gt_3,339). happensAt(walking(mws1gt_3),339).
orientation(mws1gt_1,143,340). coords(mws1gt_1,70,97,340).  visible(mws1gt_1,340). happensAt(walking(mws1gt_1),340). orientation(mws1gt_3,165,340). coords(mws1gt_3,58,150,340).  visible(mws1gt_3,340). happensAt(walking(mws1gt_3),340).
orientation(mws1gt_1,143,341). coords(mws1gt_1,71,96,341).  visible(mws1gt_1,341). happensAt(walking(mws1gt_1),341). orientation(mws1gt_3,165,341). coords(mws1gt_3,57,150,341).  visible(mws1gt_3,341). happensAt(walking(mws1gt_3),341).
orientation(mws1gt_1,143,342). coords(mws1gt_1,71,96,342).  visible(mws1gt_1,342). happensAt(walking(mws1gt_1),342). orientation(mws1gt_3,165,342). coords(mws1gt_3,56,150,342).  visible(mws1gt_3,342). happensAt(walking(mws1gt_3),342).
orientation(mws1gt_1,143,343). coords(mws1gt_1,71,95,343).  visible(mws1gt_1,343). happensAt(walking(mws1gt_1),343). orientation(mws1gt_3,165,343). coords(mws1gt_3,55,151,343).  visible(mws1gt_3,343). happensAt(walking(mws1gt_3),343).
orientation(mws1gt_1,143,344). coords(mws1gt_1,71,95,344).  visible(mws1gt_1,344). happensAt(walking(mws1gt_1),344). orientation(mws1gt_3,165,344). coords(mws1gt_3,55,151,344).  visible(mws1gt_3,344). happensAt(walking(mws1gt_3),344).
#show holdsAt/2.