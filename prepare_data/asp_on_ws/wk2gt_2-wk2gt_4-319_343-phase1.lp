
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

	orientation(wk2gt_2,141,319). coords(wk2gt_2,98,63,319).  visible(wk2gt_2,319). happensAt(inactive(wk2gt_2),319). orientation(wk2gt_4,2,319). coords(wk2gt_4,243,237,319).  visible(wk2gt_4,319). happensAt(walking(wk2gt_4),319).
orientation(wk2gt_2,141,320). coords(wk2gt_2,98,63,320).  visible(wk2gt_2,320). happensAt(inactive(wk2gt_2),320). orientation(wk2gt_4,2,320). coords(wk2gt_4,241,235,320).  visible(wk2gt_4,320). happensAt(walking(wk2gt_4),320).
orientation(wk2gt_2,141,321). coords(wk2gt_2,98,63,321).  visible(wk2gt_2,321). happensAt(inactive(wk2gt_2),321). orientation(wk2gt_4,2,321). coords(wk2gt_4,239,230,321).  visible(wk2gt_4,321). happensAt(walking(wk2gt_4),321).
orientation(wk2gt_2,141,322). coords(wk2gt_2,98,63,322).  visible(wk2gt_2,322). happensAt(inactive(wk2gt_2),322). orientation(wk2gt_4,2,322). coords(wk2gt_4,238,226,322).  visible(wk2gt_4,322). happensAt(walking(wk2gt_4),322).
orientation(wk2gt_2,141,323). coords(wk2gt_2,98,63,323).  visible(wk2gt_2,323). happensAt(inactive(wk2gt_2),323). orientation(wk2gt_4,174,323). coords(wk2gt_4,237,222,323).  visible(wk2gt_4,323). happensAt(walking(wk2gt_4),323).
orientation(wk2gt_2,141,324). coords(wk2gt_2,98,63,324).  visible(wk2gt_2,324). happensAt(inactive(wk2gt_2),324). orientation(wk2gt_4,167,324). coords(wk2gt_4,237,220,324).  visible(wk2gt_4,324). happensAt(walking(wk2gt_4),324).
orientation(wk2gt_2,141,325). coords(wk2gt_2,98,63,325).  visible(wk2gt_2,325). happensAt(inactive(wk2gt_2),325). orientation(wk2gt_4,158,325). coords(wk2gt_4,237,219,325).  visible(wk2gt_4,325). happensAt(walking(wk2gt_4),325).
orientation(wk2gt_2,141,326). coords(wk2gt_2,98,63,326).  visible(wk2gt_2,326). happensAt(inactive(wk2gt_2),326). orientation(wk2gt_4,158,326). coords(wk2gt_4,235,219,326).  visible(wk2gt_4,326). happensAt(walking(wk2gt_4),326).
orientation(wk2gt_2,141,327). coords(wk2gt_2,98,63,327).  visible(wk2gt_2,327). happensAt(inactive(wk2gt_2),327). orientation(wk2gt_4,158,327). coords(wk2gt_4,234,218,327).  visible(wk2gt_4,327). happensAt(walking(wk2gt_4),327).
orientation(wk2gt_2,141,328). coords(wk2gt_2,98,63,328).  visible(wk2gt_2,328). happensAt(inactive(wk2gt_2),328). orientation(wk2gt_4,158,328). coords(wk2gt_4,233,217,328).  visible(wk2gt_4,328). happensAt(walking(wk2gt_4),328).
orientation(wk2gt_2,141,329). coords(wk2gt_2,98,63,329).  visible(wk2gt_2,329). happensAt(inactive(wk2gt_2),329). orientation(wk2gt_4,158,329). coords(wk2gt_4,230,215,329).  visible(wk2gt_4,329). happensAt(walking(wk2gt_4),329).
orientation(wk2gt_2,141,330). coords(wk2gt_2,98,63,330).  visible(wk2gt_2,330). happensAt(inactive(wk2gt_2),330). orientation(wk2gt_4,158,330). coords(wk2gt_4,228,212,330).  visible(wk2gt_4,330). happensAt(walking(wk2gt_4),330).
orientation(wk2gt_2,141,331). coords(wk2gt_2,98,63,331).  visible(wk2gt_2,331). happensAt(inactive(wk2gt_2),331). orientation(wk2gt_4,158,331). coords(wk2gt_4,226,208,331).  visible(wk2gt_4,331). happensAt(walking(wk2gt_4),331).
orientation(wk2gt_2,141,332). coords(wk2gt_2,98,63,332).  visible(wk2gt_2,332). happensAt(inactive(wk2gt_2),332). orientation(wk2gt_4,158,332). coords(wk2gt_4,224,204,332).  visible(wk2gt_4,332). happensAt(walking(wk2gt_4),332).
orientation(wk2gt_2,141,333). coords(wk2gt_2,98,63,333).  visible(wk2gt_2,333). happensAt(inactive(wk2gt_2),333). orientation(wk2gt_4,158,333). coords(wk2gt_4,223,201,333).  visible(wk2gt_4,333). happensAt(walking(wk2gt_4),333).
orientation(wk2gt_2,141,334). coords(wk2gt_2,98,63,334).  visible(wk2gt_2,334). happensAt(inactive(wk2gt_2),334). orientation(wk2gt_4,158,334). coords(wk2gt_4,223,199,334).  visible(wk2gt_4,334). happensAt(walking(wk2gt_4),334).
orientation(wk2gt_2,141,335). coords(wk2gt_2,98,63,335).  visible(wk2gt_2,335). happensAt(inactive(wk2gt_2),335). orientation(wk2gt_4,158,335). coords(wk2gt_4,222,197,335).  visible(wk2gt_4,335). happensAt(walking(wk2gt_4),335).
orientation(wk2gt_2,141,336). coords(wk2gt_2,98,63,336).  visible(wk2gt_2,336). happensAt(inactive(wk2gt_2),336). orientation(wk2gt_4,158,336). coords(wk2gt_4,221,195,336).  visible(wk2gt_4,336). happensAt(walking(wk2gt_4),336).
orientation(wk2gt_2,141,337). coords(wk2gt_2,98,63,337).  visible(wk2gt_2,337). happensAt(inactive(wk2gt_2),337). orientation(wk2gt_4,158,337). coords(wk2gt_4,221,193,337).  visible(wk2gt_4,337). happensAt(walking(wk2gt_4),337).
orientation(wk2gt_2,141,338). coords(wk2gt_2,98,63,338).  visible(wk2gt_2,338). happensAt(inactive(wk2gt_2),338). orientation(wk2gt_4,158,338). coords(wk2gt_4,220,192,338).  visible(wk2gt_4,338). happensAt(walking(wk2gt_4),338).
orientation(wk2gt_2,141,339). coords(wk2gt_2,98,63,339).  visible(wk2gt_2,339). happensAt(inactive(wk2gt_2),339). orientation(wk2gt_4,158,339). coords(wk2gt_4,218,191,339).  visible(wk2gt_4,339). happensAt(walking(wk2gt_4),339).
orientation(wk2gt_2,141,340). coords(wk2gt_2,98,63,340).  visible(wk2gt_2,340). happensAt(inactive(wk2gt_2),340). orientation(wk2gt_4,158,340). coords(wk2gt_4,217,190,340).  visible(wk2gt_4,340). happensAt(walking(wk2gt_4),340).
orientation(wk2gt_2,141,341). coords(wk2gt_2,98,63,341).  visible(wk2gt_2,341). happensAt(inactive(wk2gt_2),341). orientation(wk2gt_4,154,341). coords(wk2gt_4,214,187,341).  visible(wk2gt_4,341). happensAt(walking(wk2gt_4),341).
orientation(wk2gt_2,141,342). coords(wk2gt_2,98,63,342).  visible(wk2gt_2,342). happensAt(inactive(wk2gt_2),342). orientation(wk2gt_4,154,342). coords(wk2gt_4,212,185,342).  visible(wk2gt_4,342). happensAt(walking(wk2gt_4),342).
orientation(wk2gt_2,141,343). coords(wk2gt_2,98,63,343).  visible(wk2gt_2,343). happensAt(inactive(wk2gt_2),343). orientation(wk2gt_4,154,343). coords(wk2gt_4,209,181,343).  visible(wk2gt_4,343). happensAt(walking(wk2gt_4),343).
#show holdsAt/2.