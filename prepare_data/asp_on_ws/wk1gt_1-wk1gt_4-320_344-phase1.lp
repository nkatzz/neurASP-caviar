
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

	orientation(wk1gt_1,151,320). coords(wk1gt_1,170,161,320).  visible(wk1gt_1,320). happensAt(walking(wk1gt_1),320). orientation(wk1gt_4,160,320). coords(wk1gt_4,52,176,320).  visible(wk1gt_4,320). happensAt(inactive(wk1gt_4),320).
orientation(wk1gt_1,158,321). coords(wk1gt_1,168,159,321).  visible(wk1gt_1,321). happensAt(walking(wk1gt_1),321). orientation(wk1gt_4,161,321). coords(wk1gt_4,52,176,321).  visible(wk1gt_4,321). happensAt(inactive(wk1gt_4),321).
orientation(wk1gt_1,158,322). coords(wk1gt_1,167,159,322).  visible(wk1gt_1,322). happensAt(walking(wk1gt_1),322). orientation(wk1gt_4,162,322). coords(wk1gt_4,52,176,322).  visible(wk1gt_4,322). happensAt(inactive(wk1gt_4),322).
orientation(wk1gt_1,158,323). coords(wk1gt_1,165,155,323).  visible(wk1gt_1,323). happensAt(walking(wk1gt_1),323). orientation(wk1gt_4,161,323). coords(wk1gt_4,52,176,323).  visible(wk1gt_4,323). happensAt(inactive(wk1gt_4),323).
orientation(wk1gt_1,158,324). coords(wk1gt_1,164,155,324).  visible(wk1gt_1,324). happensAt(walking(wk1gt_1),324). orientation(wk1gt_4,161,324). coords(wk1gt_4,52,176,324).  visible(wk1gt_4,324). happensAt(inactive(wk1gt_4),324).
orientation(wk1gt_1,158,325). coords(wk1gt_1,163,154,325).  visible(wk1gt_1,325). happensAt(walking(wk1gt_1),325). orientation(wk1gt_4,158,325). coords(wk1gt_4,52,176,325).  visible(wk1gt_4,325). happensAt(inactive(wk1gt_4),325).
orientation(wk1gt_1,158,326). coords(wk1gt_1,162,154,326).  visible(wk1gt_1,326). happensAt(walking(wk1gt_1),326). orientation(wk1gt_4,166,326). coords(wk1gt_4,52,176,326).  visible(wk1gt_4,326). happensAt(inactive(wk1gt_4),326).
orientation(wk1gt_1,150,327). coords(wk1gt_1,162,153,327).  visible(wk1gt_1,327). happensAt(walking(wk1gt_1),327). orientation(wk1gt_4,162,327). coords(wk1gt_4,52,176,327).  visible(wk1gt_4,327). happensAt(inactive(wk1gt_4),327).
orientation(wk1gt_1,150,328). coords(wk1gt_1,162,153,328).  visible(wk1gt_1,328). happensAt(walking(wk1gt_1),328). orientation(wk1gt_4,162,328). coords(wk1gt_4,52,176,328).  visible(wk1gt_4,328). happensAt(inactive(wk1gt_4),328).
orientation(wk1gt_1,150,329). coords(wk1gt_1,162,152,329).  visible(wk1gt_1,329). happensAt(walking(wk1gt_1),329). orientation(wk1gt_4,158,329). coords(wk1gt_4,52,176,329).  visible(wk1gt_4,329). happensAt(inactive(wk1gt_4),329).
orientation(wk1gt_1,148,330). coords(wk1gt_1,160,151,330).  visible(wk1gt_1,330). happensAt(walking(wk1gt_1),330). orientation(wk1gt_4,160,330). coords(wk1gt_4,52,176,330).  visible(wk1gt_4,330). happensAt(inactive(wk1gt_4),330).
orientation(wk1gt_1,148,331). coords(wk1gt_1,159,150,331).  visible(wk1gt_1,331). happensAt(walking(wk1gt_1),331). orientation(wk1gt_4,157,331). coords(wk1gt_4,52,176,331).  visible(wk1gt_4,331). happensAt(inactive(wk1gt_4),331).
orientation(wk1gt_1,148,332). coords(wk1gt_1,156,148,332).  visible(wk1gt_1,332). happensAt(walking(wk1gt_1),332). orientation(wk1gt_4,160,332). coords(wk1gt_4,52,176,332).  visible(wk1gt_4,332). happensAt(inactive(wk1gt_4),332).
orientation(wk1gt_1,148,333). coords(wk1gt_1,154,146,333).  visible(wk1gt_1,333). happensAt(walking(wk1gt_1),333). orientation(wk1gt_4,157,333). coords(wk1gt_4,52,176,333).  visible(wk1gt_4,333). happensAt(inactive(wk1gt_4),333).
orientation(wk1gt_1,148,334). coords(wk1gt_1,153,145,334).  visible(wk1gt_1,334). happensAt(walking(wk1gt_1),334). orientation(wk1gt_4,161,334). coords(wk1gt_4,52,176,334).  visible(wk1gt_4,334). happensAt(inactive(wk1gt_4),334).
orientation(wk1gt_1,148,335). coords(wk1gt_1,153,144,335).  visible(wk1gt_1,335). happensAt(walking(wk1gt_1),335). orientation(wk1gt_4,165,335). coords(wk1gt_4,52,176,335).  visible(wk1gt_4,335). happensAt(inactive(wk1gt_4),335).
orientation(wk1gt_1,153,336). coords(wk1gt_1,152,141,336).  visible(wk1gt_1,336). happensAt(walking(wk1gt_1),336). orientation(wk1gt_4,164,336). coords(wk1gt_4,52,176,336).  visible(wk1gt_4,336). happensAt(inactive(wk1gt_4),336).
orientation(wk1gt_1,153,337). coords(wk1gt_1,152,141,337).  visible(wk1gt_1,337). happensAt(walking(wk1gt_1),337). orientation(wk1gt_4,156,337). coords(wk1gt_4,52,176,337).  visible(wk1gt_4,337). happensAt(inactive(wk1gt_4),337).
orientation(wk1gt_1,153,338). coords(wk1gt_1,151,140,338).  visible(wk1gt_1,338). happensAt(walking(wk1gt_1),338). orientation(wk1gt_4,163,338). coords(wk1gt_4,52,176,338).  visible(wk1gt_4,338). happensAt(inactive(wk1gt_4),338).
orientation(wk1gt_1,153,339). coords(wk1gt_1,151,140,339).  visible(wk1gt_1,339). happensAt(walking(wk1gt_1),339). orientation(wk1gt_4,160,339). coords(wk1gt_4,52,176,339).  visible(wk1gt_4,339). happensAt(inactive(wk1gt_4),339).
orientation(wk1gt_1,149,340). coords(wk1gt_1,150,140,340).  visible(wk1gt_1,340). happensAt(walking(wk1gt_1),340). orientation(wk1gt_4,158,340). coords(wk1gt_4,52,176,340).  visible(wk1gt_4,340). happensAt(inactive(wk1gt_4),340).
orientation(wk1gt_1,149,341). coords(wk1gt_1,150,139,341).  visible(wk1gt_1,341). happensAt(walking(wk1gt_1),341). orientation(wk1gt_4,162,341). coords(wk1gt_4,52,176,341).  visible(wk1gt_4,341). happensAt(inactive(wk1gt_4),341).
orientation(wk1gt_1,149,342). coords(wk1gt_1,149,139,342).  visible(wk1gt_1,342). happensAt(walking(wk1gt_1),342). orientation(wk1gt_4,165,342). coords(wk1gt_4,52,176,342).  visible(wk1gt_4,342). happensAt(inactive(wk1gt_4),342).
orientation(wk1gt_1,149,343). coords(wk1gt_1,148,138,343).  visible(wk1gt_1,343). happensAt(walking(wk1gt_1),343). orientation(wk1gt_4,162,343). coords(wk1gt_4,52,176,343).  visible(wk1gt_4,343). happensAt(inactive(wk1gt_4),343).
orientation(wk1gt_1,149,344). coords(wk1gt_1,147,137,344).  visible(wk1gt_1,344). happensAt(walking(wk1gt_1),344). orientation(wk1gt_4,167,344). coords(wk1gt_4,52,176,344).  visible(wk1gt_4,344). happensAt(inactive(wk1gt_4),344).
#show holdsAt/2.