
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

	orientation(lb1gt_1,151,324). coords(lb1gt_1,165,170,324).  visible(lb1gt_1,324). happensAt(active(lb1gt_1),324). orientation(lb1gt_2,146,324). coords(lb1gt_2,190,150,324).  visible(lb1gt_2,324). happensAt(active(lb1gt_2),324).
orientation(lb1gt_1,151,325). coords(lb1gt_1,165,170,325).  visible(lb1gt_1,325). happensAt(active(lb1gt_1),325). orientation(lb1gt_2,146,325). coords(lb1gt_2,190,150,325).  visible(lb1gt_2,325). happensAt(active(lb1gt_2),325).
orientation(lb1gt_1,151,326). coords(lb1gt_1,165,170,326).  visible(lb1gt_1,326). happensAt(active(lb1gt_1),326). orientation(lb1gt_2,146,326). coords(lb1gt_2,190,150,326).  visible(lb1gt_2,326). happensAt(active(lb1gt_2),326).
orientation(lb1gt_1,151,327). coords(lb1gt_1,165,170,327).  visible(lb1gt_1,327). happensAt(active(lb1gt_1),327). orientation(lb1gt_2,146,327). coords(lb1gt_2,190,150,327).  visible(lb1gt_2,327). happensAt(active(lb1gt_2),327).
orientation(lb1gt_1,151,328). coords(lb1gt_1,165,170,328).  visible(lb1gt_1,328). happensAt(active(lb1gt_1),328). orientation(lb1gt_2,146,328). coords(lb1gt_2,190,150,328).  visible(lb1gt_2,328). happensAt(active(lb1gt_2),328).
orientation(lb1gt_1,151,329). coords(lb1gt_1,165,170,329).  visible(lb1gt_1,329). happensAt(active(lb1gt_1),329). orientation(lb1gt_2,146,329). coords(lb1gt_2,190,150,329).  visible(lb1gt_2,329). happensAt(active(lb1gt_2),329).
orientation(lb1gt_1,151,330). coords(lb1gt_1,165,170,330).  visible(lb1gt_1,330). happensAt(active(lb1gt_1),330). orientation(lb1gt_2,146,330). coords(lb1gt_2,190,150,330).  visible(lb1gt_2,330). happensAt(active(lb1gt_2),330).
orientation(lb1gt_1,151,331). coords(lb1gt_1,165,170,331).  visible(lb1gt_1,331). happensAt(active(lb1gt_1),331). orientation(lb1gt_2,146,331). coords(lb1gt_2,190,150,331).  visible(lb1gt_2,331). happensAt(active(lb1gt_2),331).
orientation(lb1gt_1,151,332). coords(lb1gt_1,165,169,332).  visible(lb1gt_1,332). happensAt(active(lb1gt_1),332). orientation(lb1gt_2,146,332). coords(lb1gt_2,190,150,332).  visible(lb1gt_2,332). happensAt(active(lb1gt_2),332).
orientation(lb1gt_1,151,333). coords(lb1gt_1,165,169,333).  visible(lb1gt_1,333). happensAt(active(lb1gt_1),333). orientation(lb1gt_2,146,333). coords(lb1gt_2,190,150,333).  visible(lb1gt_2,333). happensAt(active(lb1gt_2),333).
orientation(lb1gt_1,151,334). coords(lb1gt_1,165,169,334).  visible(lb1gt_1,334). happensAt(active(lb1gt_1),334). orientation(lb1gt_2,146,334). coords(lb1gt_2,190,150,334).  visible(lb1gt_2,334). happensAt(active(lb1gt_2),334).
orientation(lb1gt_1,151,335). coords(lb1gt_1,165,169,335).  visible(lb1gt_1,335). happensAt(active(lb1gt_1),335). orientation(lb1gt_2,146,335). coords(lb1gt_2,190,150,335).  visible(lb1gt_2,335). happensAt(active(lb1gt_2),335).
orientation(lb1gt_1,151,336). coords(lb1gt_1,165,169,336).  visible(lb1gt_1,336). happensAt(active(lb1gt_1),336). orientation(lb1gt_2,146,336). coords(lb1gt_2,190,150,336).  visible(lb1gt_2,336). happensAt(active(lb1gt_2),336).
orientation(lb1gt_1,151,337). coords(lb1gt_1,165,169,337).  visible(lb1gt_1,337). happensAt(active(lb1gt_1),337). orientation(lb1gt_2,146,337). coords(lb1gt_2,190,150,337).  visible(lb1gt_2,337). happensAt(active(lb1gt_2),337).
orientation(lb1gt_1,151,338). coords(lb1gt_1,165,169,338).  visible(lb1gt_1,338). happensAt(active(lb1gt_1),338). orientation(lb1gt_2,146,338). coords(lb1gt_2,190,150,338).  visible(lb1gt_2,338). happensAt(active(lb1gt_2),338).
orientation(lb1gt_1,151,339). coords(lb1gt_1,165,169,339).  visible(lb1gt_1,339). happensAt(active(lb1gt_1),339). orientation(lb1gt_2,146,339). coords(lb1gt_2,190,149,339).  visible(lb1gt_2,339). happensAt(active(lb1gt_2),339).
orientation(lb1gt_1,151,340). coords(lb1gt_1,165,169,340).  visible(lb1gt_1,340). happensAt(active(lb1gt_1),340). orientation(lb1gt_2,146,340). coords(lb1gt_2,190,149,340).  visible(lb1gt_2,340). happensAt(active(lb1gt_2),340).
orientation(lb1gt_1,151,341). coords(lb1gt_1,165,169,341).  visible(lb1gt_1,341). happensAt(active(lb1gt_1),341). orientation(lb1gt_2,146,341). coords(lb1gt_2,189,149,341).  visible(lb1gt_2,341). happensAt(active(lb1gt_2),341).
orientation(lb1gt_1,151,342). coords(lb1gt_1,165,169,342).  visible(lb1gt_1,342). happensAt(active(lb1gt_1),342). orientation(lb1gt_2,146,342). coords(lb1gt_2,189,149,342).  visible(lb1gt_2,342). happensAt(active(lb1gt_2),342).
orientation(lb1gt_1,151,343). coords(lb1gt_1,165,169,343).  visible(lb1gt_1,343). happensAt(active(lb1gt_1),343). orientation(lb1gt_2,146,343). coords(lb1gt_2,189,149,343).  visible(lb1gt_2,343). happensAt(active(lb1gt_2),343).
orientation(lb1gt_1,151,344). coords(lb1gt_1,165,169,344).  visible(lb1gt_1,344). happensAt(active(lb1gt_1),344). orientation(lb1gt_2,146,344). coords(lb1gt_2,189,148,344).  visible(lb1gt_2,344). happensAt(active(lb1gt_2),344).
orientation(lb1gt_1,151,345). coords(lb1gt_1,165,169,345).  visible(lb1gt_1,345). happensAt(active(lb1gt_1),345). orientation(lb1gt_2,146,345). coords(lb1gt_2,189,148,345).  visible(lb1gt_2,345). happensAt(active(lb1gt_2),345).
orientation(lb1gt_1,151,346). coords(lb1gt_1,165,169,346).  visible(lb1gt_1,346). happensAt(active(lb1gt_1),346). orientation(lb1gt_2,146,346). coords(lb1gt_2,189,148,346).  visible(lb1gt_2,346). happensAt(active(lb1gt_2),346).
orientation(lb1gt_1,151,347). coords(lb1gt_1,165,169,347).  visible(lb1gt_1,347). happensAt(active(lb1gt_1),347). orientation(lb1gt_2,146,347). coords(lb1gt_2,189,148,347).  visible(lb1gt_2,347). happensAt(active(lb1gt_2),347).
orientation(lb1gt_1,151,348). coords(lb1gt_1,165,169,348).  visible(lb1gt_1,348). happensAt(active(lb1gt_1),348). orientation(lb1gt_2,146,348). coords(lb1gt_2,189,148,348).  visible(lb1gt_2,348). happensAt(active(lb1gt_2),348).
#show holdsAt/2.