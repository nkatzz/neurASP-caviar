
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

	orientation(wk1gt_1,151,320). coords(wk1gt_1,170,161,320).  visible(wk1gt_1,320). happensAt(walking(wk1gt_1),320). orientation(wk1gt_5,162,320). coords(wk1gt_5,51,163,320).  visible(wk1gt_5,320). happensAt(inactive(wk1gt_5),320).
orientation(wk1gt_1,158,321). coords(wk1gt_1,168,159,321).  visible(wk1gt_1,321). happensAt(walking(wk1gt_1),321). orientation(wk1gt_5,162,321). coords(wk1gt_5,50,163,321).  visible(wk1gt_5,321). happensAt(inactive(wk1gt_5),321).
orientation(wk1gt_1,158,322). coords(wk1gt_1,167,159,322).  visible(wk1gt_1,322). happensAt(walking(wk1gt_1),322). orientation(wk1gt_5,162,322). coords(wk1gt_5,50,163,322).  visible(wk1gt_5,322). happensAt(inactive(wk1gt_5),322).
orientation(wk1gt_1,158,323). coords(wk1gt_1,165,155,323).  visible(wk1gt_1,323). happensAt(walking(wk1gt_1),323). orientation(wk1gt_5,162,323). coords(wk1gt_5,50,163,323).  visible(wk1gt_5,323). happensAt(inactive(wk1gt_5),323).
orientation(wk1gt_1,158,324). coords(wk1gt_1,164,155,324).  visible(wk1gt_1,324). happensAt(walking(wk1gt_1),324). orientation(wk1gt_5,162,324). coords(wk1gt_5,50,163,324).  visible(wk1gt_5,324). happensAt(inactive(wk1gt_5),324).
orientation(wk1gt_1,158,325). coords(wk1gt_1,163,154,325).  visible(wk1gt_1,325). happensAt(walking(wk1gt_1),325). orientation(wk1gt_5,162,325). coords(wk1gt_5,50,163,325).  visible(wk1gt_5,325). happensAt(inactive(wk1gt_5),325).
orientation(wk1gt_1,158,326). coords(wk1gt_1,162,154,326).  visible(wk1gt_1,326). happensAt(walking(wk1gt_1),326). orientation(wk1gt_5,162,326). coords(wk1gt_5,50,163,326).  visible(wk1gt_5,326). happensAt(inactive(wk1gt_5),326).
orientation(wk1gt_1,150,327). coords(wk1gt_1,162,153,327).  visible(wk1gt_1,327). happensAt(walking(wk1gt_1),327). orientation(wk1gt_5,162,327). coords(wk1gt_5,50,163,327).  visible(wk1gt_5,327). happensAt(inactive(wk1gt_5),327).
orientation(wk1gt_1,150,328). coords(wk1gt_1,162,153,328).  visible(wk1gt_1,328). happensAt(walking(wk1gt_1),328). orientation(wk1gt_5,162,328). coords(wk1gt_5,51,163,328).  visible(wk1gt_5,328). happensAt(inactive(wk1gt_5),328).
orientation(wk1gt_1,150,329). coords(wk1gt_1,162,152,329).  visible(wk1gt_1,329). happensAt(walking(wk1gt_1),329). orientation(wk1gt_5,162,329). coords(wk1gt_5,51,163,329).  visible(wk1gt_5,329). happensAt(inactive(wk1gt_5),329).
orientation(wk1gt_1,148,330). coords(wk1gt_1,160,151,330).  visible(wk1gt_1,330). happensAt(walking(wk1gt_1),330). orientation(wk1gt_5,162,330). coords(wk1gt_5,51,163,330).  visible(wk1gt_5,330). happensAt(inactive(wk1gt_5),330).
orientation(wk1gt_1,148,331). coords(wk1gt_1,159,150,331).  visible(wk1gt_1,331). happensAt(walking(wk1gt_1),331). orientation(wk1gt_5,162,331). coords(wk1gt_5,51,163,331).  visible(wk1gt_5,331). happensAt(inactive(wk1gt_5),331).
orientation(wk1gt_1,148,332). coords(wk1gt_1,156,148,332).  visible(wk1gt_1,332). happensAt(walking(wk1gt_1),332). orientation(wk1gt_5,162,332). coords(wk1gt_5,50,163,332).  visible(wk1gt_5,332). happensAt(inactive(wk1gt_5),332).
orientation(wk1gt_1,148,333). coords(wk1gt_1,154,146,333).  visible(wk1gt_1,333). happensAt(walking(wk1gt_1),333). orientation(wk1gt_5,162,333). coords(wk1gt_5,50,163,333).  visible(wk1gt_5,333). happensAt(inactive(wk1gt_5),333).
orientation(wk1gt_1,148,334). coords(wk1gt_1,153,145,334).  visible(wk1gt_1,334). happensAt(walking(wk1gt_1),334). orientation(wk1gt_5,162,334). coords(wk1gt_5,50,163,334).  visible(wk1gt_5,334). happensAt(inactive(wk1gt_5),334).
orientation(wk1gt_1,148,335). coords(wk1gt_1,153,144,335).  visible(wk1gt_1,335). happensAt(walking(wk1gt_1),335). orientation(wk1gt_5,162,335). coords(wk1gt_5,50,163,335).  visible(wk1gt_5,335). happensAt(inactive(wk1gt_5),335).
orientation(wk1gt_1,153,336). coords(wk1gt_1,152,141,336).  visible(wk1gt_1,336). happensAt(walking(wk1gt_1),336). orientation(wk1gt_5,162,336). coords(wk1gt_5,50,163,336).  visible(wk1gt_5,336). happensAt(inactive(wk1gt_5),336).
orientation(wk1gt_1,153,337). coords(wk1gt_1,152,141,337).  visible(wk1gt_1,337). happensAt(walking(wk1gt_1),337). orientation(wk1gt_5,162,337). coords(wk1gt_5,50,163,337).  visible(wk1gt_5,337). happensAt(inactive(wk1gt_5),337).
orientation(wk1gt_1,153,338). coords(wk1gt_1,151,140,338).  visible(wk1gt_1,338). happensAt(walking(wk1gt_1),338). orientation(wk1gt_5,162,338). coords(wk1gt_5,50,163,338).  visible(wk1gt_5,338). happensAt(inactive(wk1gt_5),338).
orientation(wk1gt_1,153,339). coords(wk1gt_1,151,140,339).  visible(wk1gt_1,339). happensAt(walking(wk1gt_1),339). orientation(wk1gt_5,162,339). coords(wk1gt_5,50,163,339).  visible(wk1gt_5,339). happensAt(inactive(wk1gt_5),339).
orientation(wk1gt_1,149,340). coords(wk1gt_1,150,140,340).  visible(wk1gt_1,340). happensAt(walking(wk1gt_1),340). orientation(wk1gt_5,162,340). coords(wk1gt_5,50,163,340).  visible(wk1gt_5,340). happensAt(inactive(wk1gt_5),340).
orientation(wk1gt_1,149,341). coords(wk1gt_1,150,139,341).  visible(wk1gt_1,341). happensAt(walking(wk1gt_1),341). orientation(wk1gt_5,162,341). coords(wk1gt_5,50,163,341).  visible(wk1gt_5,341). happensAt(inactive(wk1gt_5),341).
orientation(wk1gt_1,149,342). coords(wk1gt_1,149,139,342).  visible(wk1gt_1,342). happensAt(walking(wk1gt_1),342). orientation(wk1gt_5,162,342). coords(wk1gt_5,50,163,342).  visible(wk1gt_5,342). happensAt(inactive(wk1gt_5),342).
orientation(wk1gt_1,149,343). coords(wk1gt_1,148,138,343).  visible(wk1gt_1,343). happensAt(walking(wk1gt_1),343). orientation(wk1gt_5,162,343). coords(wk1gt_5,50,163,343).  visible(wk1gt_5,343). happensAt(inactive(wk1gt_5),343).
orientation(wk1gt_1,149,344). coords(wk1gt_1,147,137,344).  visible(wk1gt_1,344). happensAt(walking(wk1gt_1),344). orientation(wk1gt_5,162,344). coords(wk1gt_5,50,163,344).  visible(wk1gt_5,344). happensAt(inactive(wk1gt_5),344).
#show holdsAt/2.