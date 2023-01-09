
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

	orientation(mws1gt_3,165,324). coords(mws1gt_3,63,141,324).  visible(mws1gt_3,324). happensAt(walking(mws1gt_3),324). orientation(mws1gt_6,146,324). coords(mws1gt_6,33,176,324).  visible(mws1gt_6,324). happensAt(walking(mws1gt_6),324).
orientation(mws1gt_3,165,325). coords(mws1gt_3,63,141,325).  visible(mws1gt_3,325). happensAt(walking(mws1gt_3),325). orientation(mws1gt_6,146,325). coords(mws1gt_6,33,176,325).  visible(mws1gt_6,325). happensAt(walking(mws1gt_6),325).
orientation(mws1gt_3,165,326). coords(mws1gt_3,63,142,326).  visible(mws1gt_3,326). happensAt(walking(mws1gt_3),326). orientation(mws1gt_6,146,326). coords(mws1gt_6,33,177,326).  visible(mws1gt_6,326). happensAt(walking(mws1gt_6),326).
orientation(mws1gt_3,165,327). coords(mws1gt_3,62,143,327).  visible(mws1gt_3,327). happensAt(walking(mws1gt_3),327). orientation(mws1gt_6,146,327). coords(mws1gt_6,33,178,327).  visible(mws1gt_6,327). happensAt(walking(mws1gt_6),327).
orientation(mws1gt_3,165,328). coords(mws1gt_3,61,143,328).  visible(mws1gt_3,328). happensAt(walking(mws1gt_3),328). orientation(mws1gt_6,146,328). coords(mws1gt_6,33,178,328).  visible(mws1gt_6,328). happensAt(walking(mws1gt_6),328).
orientation(mws1gt_3,165,329). coords(mws1gt_3,61,143,329).  visible(mws1gt_3,329). happensAt(walking(mws1gt_3),329). orientation(mws1gt_6,146,329). coords(mws1gt_6,33,178,329).  visible(mws1gt_6,329). happensAt(walking(mws1gt_6),329).
orientation(mws1gt_3,165,330). coords(mws1gt_3,60,144,330).  visible(mws1gt_3,330). happensAt(walking(mws1gt_3),330). orientation(mws1gt_6,146,330). coords(mws1gt_6,33,178,330).  visible(mws1gt_6,330). happensAt(walking(mws1gt_6),330).
orientation(mws1gt_3,165,331). coords(mws1gt_3,60,146,331).  visible(mws1gt_3,331). happensAt(walking(mws1gt_3),331). orientation(mws1gt_6,146,331). coords(mws1gt_6,33,179,331).  visible(mws1gt_6,331). happensAt(walking(mws1gt_6),331).
orientation(mws1gt_3,165,332). coords(mws1gt_3,60,146,332).  visible(mws1gt_3,332). happensAt(walking(mws1gt_3),332). orientation(mws1gt_6,146,332). coords(mws1gt_6,33,179,332).  visible(mws1gt_6,332). happensAt(walking(mws1gt_6),332).
orientation(mws1gt_3,165,333). coords(mws1gt_3,60,147,333).  visible(mws1gt_3,333). happensAt(walking(mws1gt_3),333). orientation(mws1gt_6,146,333). coords(mws1gt_6,33,179,333).  visible(mws1gt_6,333). happensAt(walking(mws1gt_6),333).
orientation(mws1gt_3,165,334). coords(mws1gt_3,60,147,334).  visible(mws1gt_3,334). happensAt(walking(mws1gt_3),334). orientation(mws1gt_6,146,334). coords(mws1gt_6,33,180,334).  visible(mws1gt_6,334). happensAt(walking(mws1gt_6),334).
orientation(mws1gt_3,165,335). coords(mws1gt_3,60,148,335).  visible(mws1gt_3,335). happensAt(walking(mws1gt_3),335). orientation(mws1gt_6,146,335). coords(mws1gt_6,32,182,335).  visible(mws1gt_6,335). happensAt(walking(mws1gt_6),335).
orientation(mws1gt_3,165,336). coords(mws1gt_3,60,149,336).  visible(mws1gt_3,336). happensAt(walking(mws1gt_3),336). orientation(mws1gt_6,146,336). coords(mws1gt_6,32,183,336).  visible(mws1gt_6,336). happensAt(walking(mws1gt_6),336).
orientation(mws1gt_3,165,337). coords(mws1gt_3,59,149,337).  visible(mws1gt_3,337). happensAt(walking(mws1gt_3),337). orientation(mws1gt_6,146,337). coords(mws1gt_6,32,184,337).  visible(mws1gt_6,337). happensAt(walking(mws1gt_6),337).
orientation(mws1gt_3,165,338). coords(mws1gt_3,59,149,338).  visible(mws1gt_3,338). happensAt(walking(mws1gt_3),338). orientation(mws1gt_6,146,338). coords(mws1gt_6,32,185,338).  visible(mws1gt_6,338). happensAt(walking(mws1gt_6),338).
orientation(mws1gt_3,165,339). coords(mws1gt_3,59,149,339).  visible(mws1gt_3,339). happensAt(walking(mws1gt_3),339). orientation(mws1gt_6,146,339). coords(mws1gt_6,32,186,339).  visible(mws1gt_6,339). happensAt(walking(mws1gt_6),339).
orientation(mws1gt_3,165,340). coords(mws1gt_3,58,150,340).  visible(mws1gt_3,340). happensAt(walking(mws1gt_3),340). orientation(mws1gt_6,160,340). coords(mws1gt_6,32,187,340).  visible(mws1gt_6,340). happensAt(walking(mws1gt_6),340).
orientation(mws1gt_3,165,341). coords(mws1gt_3,57,150,341).  visible(mws1gt_3,341). happensAt(walking(mws1gt_3),341). orientation(mws1gt_6,160,341). coords(mws1gt_6,32,187,341).  visible(mws1gt_6,341). happensAt(walking(mws1gt_6),341).
orientation(mws1gt_3,165,342). coords(mws1gt_3,56,150,342).  visible(mws1gt_3,342). happensAt(walking(mws1gt_3),342). orientation(mws1gt_6,160,342). coords(mws1gt_6,32,188,342).  visible(mws1gt_6,342). happensAt(walking(mws1gt_6),342).
orientation(mws1gt_3,165,343). coords(mws1gt_3,55,151,343).  visible(mws1gt_3,343). happensAt(walking(mws1gt_3),343). orientation(mws1gt_6,160,343). coords(mws1gt_6,32,189,343).  visible(mws1gt_6,343). happensAt(walking(mws1gt_6),343).
orientation(mws1gt_3,165,344). coords(mws1gt_3,55,151,344).  visible(mws1gt_3,344). happensAt(walking(mws1gt_3),344). orientation(mws1gt_6,160,344). coords(mws1gt_6,32,189,344).  visible(mws1gt_6,344). happensAt(walking(mws1gt_6),344).
orientation(mws1gt_3,165,345). coords(mws1gt_3,55,151,345).  visible(mws1gt_3,345). happensAt(walking(mws1gt_3),345). orientation(mws1gt_6,160,345). coords(mws1gt_6,32,189,345).  visible(mws1gt_6,345). happensAt(walking(mws1gt_6),345).
orientation(mws1gt_3,165,346). coords(mws1gt_3,54,152,346).  visible(mws1gt_3,346). happensAt(walking(mws1gt_3),346). orientation(mws1gt_6,160,346). coords(mws1gt_6,32,190,346).  visible(mws1gt_6,346). happensAt(walking(mws1gt_6),346).
orientation(mws1gt_3,165,347). coords(mws1gt_3,54,152,347).  visible(mws1gt_3,347). happensAt(walking(mws1gt_3),347). orientation(mws1gt_6,160,347). coords(mws1gt_6,32,191,347).  visible(mws1gt_6,347). happensAt(walking(mws1gt_6),347).
orientation(mws1gt_3,165,348). coords(mws1gt_3,54,152,348).  visible(mws1gt_3,348). happensAt(walking(mws1gt_3),348). orientation(mws1gt_6,160,348). coords(mws1gt_6,32,192,348).  visible(mws1gt_6,348). happensAt(walking(mws1gt_6),348).
#show holdsAt/2.