
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

	orientation(mws1gt_6,146,315). coords(mws1gt_6,35,167,315).  visible(mws1gt_6,315). happensAt(walking(mws1gt_6),315). orientation(mws1gt_7,14,315). coords(mws1gt_7,33,206,315).  visible(mws1gt_7,315). happensAt(walking(mws1gt_7),315).
orientation(mws1gt_6,146,316). coords(mws1gt_6,35,167,316).  visible(mws1gt_6,316). happensAt(walking(mws1gt_6),316). orientation(mws1gt_7,14,316). coords(mws1gt_7,33,206,316).  visible(mws1gt_7,316). happensAt(walking(mws1gt_7),316).
orientation(mws1gt_6,146,317). coords(mws1gt_6,35,167,317).  visible(mws1gt_6,317). happensAt(walking(mws1gt_6),317). orientation(mws1gt_7,14,317). coords(mws1gt_7,34,205,317).  visible(mws1gt_7,317). happensAt(walking(mws1gt_7),317).
orientation(mws1gt_6,146,318). coords(mws1gt_6,35,168,318).  visible(mws1gt_6,318). happensAt(walking(mws1gt_6),318). orientation(mws1gt_7,14,318). coords(mws1gt_7,36,205,318).  visible(mws1gt_7,318). happensAt(walking(mws1gt_7),318).
orientation(mws1gt_6,146,319). coords(mws1gt_6,35,169,319).  visible(mws1gt_6,319). happensAt(walking(mws1gt_6),319). orientation(mws1gt_7,14,319). coords(mws1gt_7,36,204,319).  visible(mws1gt_7,319). happensAt(walking(mws1gt_7),319).
orientation(mws1gt_6,146,320). coords(mws1gt_6,35,170,320).  visible(mws1gt_6,320). happensAt(walking(mws1gt_6),320). orientation(mws1gt_7,14,320). coords(mws1gt_7,36,204,320).  visible(mws1gt_7,320). happensAt(walking(mws1gt_7),320).
orientation(mws1gt_6,146,321). coords(mws1gt_6,34,172,321).  visible(mws1gt_6,321). happensAt(walking(mws1gt_6),321). orientation(mws1gt_7,14,321). coords(mws1gt_7,36,204,321).  visible(mws1gt_7,321). happensAt(walking(mws1gt_7),321).
orientation(mws1gt_6,146,322). coords(mws1gt_6,33,174,322).  visible(mws1gt_6,322). happensAt(walking(mws1gt_6),322). orientation(mws1gt_7,14,322). coords(mws1gt_7,37,202,322).  visible(mws1gt_7,322). happensAt(walking(mws1gt_7),322).
orientation(mws1gt_6,146,323). coords(mws1gt_6,33,176,323).  visible(mws1gt_6,323). happensAt(walking(mws1gt_6),323). orientation(mws1gt_7,14,323). coords(mws1gt_7,37,201,323).  visible(mws1gt_7,323). happensAt(walking(mws1gt_7),323).
orientation(mws1gt_6,146,324). coords(mws1gt_6,33,176,324).  visible(mws1gt_6,324). happensAt(walking(mws1gt_6),324). orientation(mws1gt_7,0,324). coords(mws1gt_7,38,200,324).  visible(mws1gt_7,324). happensAt(walking(mws1gt_7),324).
orientation(mws1gt_6,146,325). coords(mws1gt_6,33,176,325).  visible(mws1gt_6,325). happensAt(walking(mws1gt_6),325). orientation(mws1gt_7,0,325). coords(mws1gt_7,38,199,325).  visible(mws1gt_7,325). happensAt(walking(mws1gt_7),325).
orientation(mws1gt_6,146,326). coords(mws1gt_6,33,177,326).  visible(mws1gt_6,326). happensAt(walking(mws1gt_6),326). orientation(mws1gt_7,170,326). coords(mws1gt_7,38,199,326).  visible(mws1gt_7,326). happensAt(walking(mws1gt_7),326).
orientation(mws1gt_6,146,327). coords(mws1gt_6,33,178,327).  visible(mws1gt_6,327). happensAt(walking(mws1gt_6),327). orientation(mws1gt_7,162,327). coords(mws1gt_7,38,199,327).  visible(mws1gt_7,327). happensAt(walking(mws1gt_7),327).
orientation(mws1gt_6,146,328). coords(mws1gt_6,33,178,328).  visible(mws1gt_6,328). happensAt(walking(mws1gt_6),328). orientation(mws1gt_7,167,328). coords(mws1gt_7,38,197,328).  visible(mws1gt_7,328). happensAt(walking(mws1gt_7),328).
orientation(mws1gt_6,146,329). coords(mws1gt_6,33,178,329).  visible(mws1gt_6,329). happensAt(walking(mws1gt_6),329). orientation(mws1gt_7,167,329). coords(mws1gt_7,39,197,329).  visible(mws1gt_7,329). happensAt(walking(mws1gt_7),329).
orientation(mws1gt_6,146,330). coords(mws1gt_6,33,178,330).  visible(mws1gt_6,330). happensAt(walking(mws1gt_6),330). orientation(mws1gt_7,167,330). coords(mws1gt_7,39,197,330).  visible(mws1gt_7,330). happensAt(walking(mws1gt_7),330).
orientation(mws1gt_6,146,331). coords(mws1gt_6,33,179,331).  visible(mws1gt_6,331). happensAt(walking(mws1gt_6),331). orientation(mws1gt_7,167,331). coords(mws1gt_7,40,197,331).  visible(mws1gt_7,331). happensAt(walking(mws1gt_7),331).
orientation(mws1gt_6,146,332). coords(mws1gt_6,33,179,332).  visible(mws1gt_6,332). happensAt(walking(mws1gt_6),332). orientation(mws1gt_7,167,332). coords(mws1gt_7,40,196,332).  visible(mws1gt_7,332). happensAt(walking(mws1gt_7),332).
orientation(mws1gt_6,146,333). coords(mws1gt_6,33,179,333).  visible(mws1gt_6,333). happensAt(walking(mws1gt_6),333). orientation(mws1gt_7,167,333). coords(mws1gt_7,41,196,333).  visible(mws1gt_7,333). happensAt(walking(mws1gt_7),333).
orientation(mws1gt_6,146,334). coords(mws1gt_6,33,180,334).  visible(mws1gt_6,334). happensAt(walking(mws1gt_6),334). orientation(mws1gt_7,167,334). coords(mws1gt_7,41,196,334).  visible(mws1gt_7,334). happensAt(walking(mws1gt_7),334).
orientation(mws1gt_6,146,335). coords(mws1gt_6,32,182,335).  visible(mws1gt_6,335). happensAt(walking(mws1gt_6),335). orientation(mws1gt_7,167,335). coords(mws1gt_7,41,195,335).  visible(mws1gt_7,335). happensAt(walking(mws1gt_7),335).
orientation(mws1gt_6,146,336). coords(mws1gt_6,32,183,336).  visible(mws1gt_6,336). happensAt(walking(mws1gt_6),336). orientation(mws1gt_7,167,336). coords(mws1gt_7,41,195,336).  visible(mws1gt_7,336). happensAt(walking(mws1gt_7),336).
orientation(mws1gt_6,146,337). coords(mws1gt_6,32,184,337).  visible(mws1gt_6,337). happensAt(walking(mws1gt_6),337). orientation(mws1gt_7,167,337). coords(mws1gt_7,41,194,337).  visible(mws1gt_7,337). happensAt(walking(mws1gt_7),337).
orientation(mws1gt_6,146,338). coords(mws1gt_6,32,185,338).  visible(mws1gt_6,338). happensAt(walking(mws1gt_6),338). orientation(mws1gt_7,167,338). coords(mws1gt_7,41,193,338).  visible(mws1gt_7,338). happensAt(walking(mws1gt_7),338).
orientation(mws1gt_6,146,339). coords(mws1gt_6,32,186,339).  visible(mws1gt_6,339). happensAt(walking(mws1gt_6),339). orientation(mws1gt_7,167,339). coords(mws1gt_7,41,193,339).  visible(mws1gt_7,339). happensAt(walking(mws1gt_7),339).
#show holdsAt/2.