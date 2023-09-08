
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

	orientation(fcgt_2,146,330). coords(fcgt_2,91,125,330).  visible(fcgt_2,330). happensAt(active(fcgt_2),330). orientation(fcgt_4,166,330). coords(fcgt_4,82,206,330).  visible(fcgt_4,330). happensAt(inactive(fcgt_4),330).
orientation(fcgt_2,146,331). coords(fcgt_2,91,124,331).  visible(fcgt_2,331). happensAt(active(fcgt_2),331). orientation(fcgt_4,166,331). coords(fcgt_4,82,206,331).  visible(fcgt_4,331). happensAt(inactive(fcgt_4),331).
orientation(fcgt_2,146,332). coords(fcgt_2,91,124,332).  visible(fcgt_2,332). happensAt(active(fcgt_2),332). orientation(fcgt_4,166,332). coords(fcgt_4,82,206,332).  visible(fcgt_4,332). happensAt(inactive(fcgt_4),332).
orientation(fcgt_2,146,333). coords(fcgt_2,92,124,333).  visible(fcgt_2,333). happensAt(active(fcgt_2),333). orientation(fcgt_4,166,333). coords(fcgt_4,82,206,333).  visible(fcgt_4,333). happensAt(inactive(fcgt_4),333).
orientation(fcgt_2,146,334). coords(fcgt_2,93,123,334).  visible(fcgt_2,334). happensAt(active(fcgt_2),334). orientation(fcgt_4,166,334). coords(fcgt_4,82,206,334).  visible(fcgt_4,334). happensAt(inactive(fcgt_4),334).
orientation(fcgt_2,146,335). coords(fcgt_2,94,123,335).  visible(fcgt_2,335). happensAt(active(fcgt_2),335). orientation(fcgt_4,166,335). coords(fcgt_4,82,206,335).  visible(fcgt_4,335). happensAt(inactive(fcgt_4),335).
orientation(fcgt_2,155,336). coords(fcgt_2,94,121,336).  visible(fcgt_2,336). happensAt(active(fcgt_2),336). orientation(fcgt_4,166,336). coords(fcgt_4,82,206,336).  visible(fcgt_4,336). happensAt(inactive(fcgt_4),336).
orientation(fcgt_2,155,337). coords(fcgt_2,94,120,337).  visible(fcgt_2,337). happensAt(active(fcgt_2),337). orientation(fcgt_4,166,337). coords(fcgt_4,82,206,337).  visible(fcgt_4,337). happensAt(inactive(fcgt_4),337).
orientation(fcgt_2,155,338). coords(fcgt_2,94,120,338).  visible(fcgt_2,338). happensAt(active(fcgt_2),338). orientation(fcgt_4,166,338). coords(fcgt_4,82,206,338).  visible(fcgt_4,338). happensAt(inactive(fcgt_4),338).
orientation(fcgt_2,155,339). coords(fcgt_2,94,119,339).  visible(fcgt_2,339). happensAt(active(fcgt_2),339). orientation(fcgt_4,166,339). coords(fcgt_4,82,206,339).  visible(fcgt_4,339). happensAt(inactive(fcgt_4),339).
orientation(fcgt_2,155,340). coords(fcgt_2,94,117,340).  visible(fcgt_2,340). happensAt(active(fcgt_2),340). orientation(fcgt_4,166,340). coords(fcgt_4,82,206,340).  visible(fcgt_4,340). happensAt(inactive(fcgt_4),340).
orientation(fcgt_2,155,341). coords(fcgt_2,94,117,341).  visible(fcgt_2,341). happensAt(active(fcgt_2),341). orientation(fcgt_4,166,341). coords(fcgt_4,82,206,341).  visible(fcgt_4,341). happensAt(inactive(fcgt_4),341).
orientation(fcgt_2,155,342). coords(fcgt_2,94,116,342).  visible(fcgt_2,342). happensAt(active(fcgt_2),342). orientation(fcgt_4,166,342). coords(fcgt_4,82,206,342).  visible(fcgt_4,342). happensAt(inactive(fcgt_4),342).
orientation(fcgt_2,155,343). coords(fcgt_2,94,115,343).  visible(fcgt_2,343). happensAt(active(fcgt_2),343). orientation(fcgt_4,166,343). coords(fcgt_4,82,206,343).  visible(fcgt_4,343). happensAt(inactive(fcgt_4),343).
orientation(fcgt_2,155,344). coords(fcgt_2,94,115,344).  visible(fcgt_2,344). happensAt(active(fcgt_2),344). orientation(fcgt_4,166,344). coords(fcgt_4,82,206,344).  visible(fcgt_4,344). happensAt(inactive(fcgt_4),344).
orientation(fcgt_2,155,345). coords(fcgt_2,94,114,345).  visible(fcgt_2,345). happensAt(active(fcgt_2),345). orientation(fcgt_4,166,345). coords(fcgt_4,82,206,345).  visible(fcgt_4,345). happensAt(inactive(fcgt_4),345).
orientation(fcgt_2,155,346). coords(fcgt_2,93,114,346).  visible(fcgt_2,346). happensAt(active(fcgt_2),346). orientation(fcgt_4,166,346). coords(fcgt_4,82,206,346).  visible(fcgt_4,346). happensAt(inactive(fcgt_4),346).
orientation(fcgt_2,155,347). coords(fcgt_2,94,114,347).  visible(fcgt_2,347). happensAt(active(fcgt_2),347). orientation(fcgt_4,166,347). coords(fcgt_4,82,206,347).  visible(fcgt_4,347). happensAt(inactive(fcgt_4),347).
orientation(fcgt_2,155,348). coords(fcgt_2,94,114,348).  visible(fcgt_2,348). happensAt(active(fcgt_2),348). orientation(fcgt_4,166,348). coords(fcgt_4,82,206,348).  visible(fcgt_4,348). happensAt(inactive(fcgt_4),348).
orientation(fcgt_2,155,349). coords(fcgt_2,93,113,349).  visible(fcgt_2,349). happensAt(active(fcgt_2),349). orientation(fcgt_4,166,349). coords(fcgt_4,82,206,349).  visible(fcgt_4,349). happensAt(inactive(fcgt_4),349).
orientation(fcgt_2,154,350). coords(fcgt_2,93,112,350).  visible(fcgt_2,350). happensAt(active(fcgt_2),350). orientation(fcgt_4,166,350). coords(fcgt_4,82,206,350).  visible(fcgt_4,350). happensAt(inactive(fcgt_4),350).
orientation(fcgt_2,149,351). coords(fcgt_2,92,111,351).  visible(fcgt_2,351). happensAt(active(fcgt_2),351). orientation(fcgt_4,166,351). coords(fcgt_4,82,206,351).  visible(fcgt_4,351). happensAt(inactive(fcgt_4),351).
orientation(fcgt_2,147,352). coords(fcgt_2,92,111,352).  visible(fcgt_2,352). happensAt(active(fcgt_2),352). orientation(fcgt_4,166,352). coords(fcgt_4,82,206,352).  visible(fcgt_4,352). happensAt(inactive(fcgt_4),352).
orientation(fcgt_2,148,353). coords(fcgt_2,92,110,353).  visible(fcgt_2,353). happensAt(active(fcgt_2),353). orientation(fcgt_4,166,353). coords(fcgt_4,82,206,353).  visible(fcgt_4,353). happensAt(inactive(fcgt_4),353).
orientation(fcgt_2,146,354). coords(fcgt_2,92,110,354).  visible(fcgt_2,354). happensAt(active(fcgt_2),354). orientation(fcgt_4,166,354). coords(fcgt_4,82,206,354).  visible(fcgt_4,354). happensAt(inactive(fcgt_4),354).
#show holdsAt/2.