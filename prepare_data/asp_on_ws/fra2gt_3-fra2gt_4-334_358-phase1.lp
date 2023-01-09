
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

	orientation(fra2gt_3,138,334). coords(fra2gt_3,89,114,334).  visible(fra2gt_3,334). happensAt(walking(fra2gt_3),334). orientation(fra2gt_4,155,334). coords(fra2gt_4,99,116,334).  visible(fra2gt_4,334). happensAt(walking(fra2gt_4),334).
orientation(fra2gt_3,138,335). coords(fra2gt_3,89,113,335).  visible(fra2gt_3,335). happensAt(walking(fra2gt_3),335). orientation(fra2gt_4,155,335). coords(fra2gt_4,99,116,335).  visible(fra2gt_4,335). happensAt(walking(fra2gt_4),335).
orientation(fra2gt_3,138,336). coords(fra2gt_3,89,109,336).  visible(fra2gt_3,336). happensAt(walking(fra2gt_3),336). orientation(fra2gt_4,155,336). coords(fra2gt_4,98,114,336).  visible(fra2gt_4,336). happensAt(walking(fra2gt_4),336).
orientation(fra2gt_3,138,337). coords(fra2gt_3,89,108,337).  visible(fra2gt_3,337). happensAt(walking(fra2gt_3),337). orientation(fra2gt_4,155,337). coords(fra2gt_4,97,112,337).  visible(fra2gt_4,337). happensAt(walking(fra2gt_4),337).
orientation(fra2gt_3,138,338). coords(fra2gt_3,89,108,338).  visible(fra2gt_3,338). happensAt(walking(fra2gt_3),338). orientation(fra2gt_4,155,338). coords(fra2gt_4,98,110,338).  visible(fra2gt_4,338). happensAt(walking(fra2gt_4),338).
orientation(fra2gt_3,138,339). coords(fra2gt_3,91,109,339).  visible(fra2gt_3,339). happensAt(walking(fra2gt_3),339). orientation(fra2gt_4,155,339). coords(fra2gt_4,98,110,339).  visible(fra2gt_4,339). happensAt(walking(fra2gt_4),339).
orientation(fra2gt_3,138,340). coords(fra2gt_3,91,109,340).  visible(fra2gt_3,340). happensAt(walking(fra2gt_3),340). orientation(fra2gt_4,155,340). coords(fra2gt_4,98,111,340).  visible(fra2gt_4,340). happensAt(walking(fra2gt_4),340).
orientation(fra2gt_3,138,341). coords(fra2gt_3,91,109,341).  visible(fra2gt_3,341). happensAt(walking(fra2gt_3),341). orientation(fra2gt_4,155,341). coords(fra2gt_4,97,113,341).  visible(fra2gt_4,341). happensAt(walking(fra2gt_4),341).
orientation(fra2gt_3,138,342). coords(fra2gt_3,91,110,342).  visible(fra2gt_3,342). happensAt(walking(fra2gt_3),342). orientation(fra2gt_4,155,342). coords(fra2gt_4,97,115,342).  visible(fra2gt_4,342). happensAt(walking(fra2gt_4),342).
orientation(fra2gt_3,138,343). coords(fra2gt_3,91,110,343).  visible(fra2gt_3,343). happensAt(walking(fra2gt_3),343). orientation(fra2gt_4,155,343). coords(fra2gt_4,97,117,343).  visible(fra2gt_4,343). happensAt(walking(fra2gt_4),343).
orientation(fra2gt_3,138,344). coords(fra2gt_3,91,110,344).  visible(fra2gt_3,344). happensAt(walking(fra2gt_3),344). orientation(fra2gt_4,155,344). coords(fra2gt_4,97,118,344).  visible(fra2gt_4,344). happensAt(walking(fra2gt_4),344).
orientation(fra2gt_3,138,345). coords(fra2gt_3,91,111,345).  visible(fra2gt_3,345). happensAt(walking(fra2gt_3),345). orientation(fra2gt_4,155,345). coords(fra2gt_4,97,119,345).  visible(fra2gt_4,345). happensAt(walking(fra2gt_4),345).
orientation(fra2gt_3,138,346). coords(fra2gt_3,90,111,346).  visible(fra2gt_3,346). happensAt(walking(fra2gt_3),346). orientation(fra2gt_4,155,346). coords(fra2gt_4,98,119,346).  visible(fra2gt_4,346). happensAt(walking(fra2gt_4),346).
orientation(fra2gt_3,138,347). coords(fra2gt_3,90,111,347).  visible(fra2gt_3,347). happensAt(walking(fra2gt_3),347). orientation(fra2gt_4,155,347). coords(fra2gt_4,97,118,347).  visible(fra2gt_4,347). happensAt(walking(fra2gt_4),347).
orientation(fra2gt_3,138,348). coords(fra2gt_3,89,111,348).  visible(fra2gt_3,348). happensAt(walking(fra2gt_3),348). orientation(fra2gt_4,155,348). coords(fra2gt_4,95,117,348).  visible(fra2gt_4,348). happensAt(walking(fra2gt_4),348).
orientation(fra2gt_3,138,349). coords(fra2gt_3,89,111,349).  visible(fra2gt_3,349). happensAt(walking(fra2gt_3),349). orientation(fra2gt_4,155,349). coords(fra2gt_4,95,116,349).  visible(fra2gt_4,349). happensAt(walking(fra2gt_4),349).
orientation(fra2gt_3,138,350). coords(fra2gt_3,89,111,350).  visible(fra2gt_3,350). happensAt(walking(fra2gt_3),350). orientation(fra2gt_4,155,350). coords(fra2gt_4,95,116,350).  visible(fra2gt_4,350). happensAt(walking(fra2gt_4),350).
orientation(fra2gt_3,138,351). coords(fra2gt_3,88,111,351).  visible(fra2gt_3,351). happensAt(walking(fra2gt_3),351). orientation(fra2gt_4,155,351). coords(fra2gt_4,96,116,351).  visible(fra2gt_4,351). happensAt(walking(fra2gt_4),351).
orientation(fra2gt_3,138,352). coords(fra2gt_3,88,112,352).  visible(fra2gt_3,352). happensAt(walking(fra2gt_3),352). orientation(fra2gt_4,155,352). coords(fra2gt_4,95,116,352).  visible(fra2gt_4,352). happensAt(walking(fra2gt_4),352).
orientation(fra2gt_3,138,353). coords(fra2gt_3,85,112,353).  visible(fra2gt_3,353). happensAt(walking(fra2gt_3),353). orientation(fra2gt_4,155,353). coords(fra2gt_4,95,116,353).  visible(fra2gt_4,353). happensAt(walking(fra2gt_4),353).
orientation(fra2gt_3,138,354). coords(fra2gt_3,85,113,354).  visible(fra2gt_3,354). happensAt(walking(fra2gt_3),354). orientation(fra2gt_4,155,354). coords(fra2gt_4,97,115,354).  visible(fra2gt_4,354). happensAt(walking(fra2gt_4),354).
orientation(fra2gt_3,138,355). coords(fra2gt_3,83,114,355).  visible(fra2gt_3,355). happensAt(walking(fra2gt_3),355). orientation(fra2gt_4,155,355). coords(fra2gt_4,97,116,355).  visible(fra2gt_4,355). happensAt(walking(fra2gt_4),355).
orientation(fra2gt_3,138,356). coords(fra2gt_3,82,114,356).  visible(fra2gt_3,356). happensAt(walking(fra2gt_3),356). orientation(fra2gt_4,155,356). coords(fra2gt_4,95,116,356).  visible(fra2gt_4,356). happensAt(walking(fra2gt_4),356).
orientation(fra2gt_3,138,357). coords(fra2gt_3,83,114,357).  visible(fra2gt_3,357). happensAt(walking(fra2gt_3),357). orientation(fra2gt_4,155,357). coords(fra2gt_4,95,118,357).  visible(fra2gt_4,357). happensAt(walking(fra2gt_4),357).
orientation(fra2gt_3,138,358). coords(fra2gt_3,83,114,358).  visible(fra2gt_3,358). happensAt(walking(fra2gt_3),358). orientation(fra2gt_4,155,358). coords(fra2gt_4,95,117,358).  visible(fra2gt_4,358). happensAt(walking(fra2gt_4),358).
#show holdsAt/2.