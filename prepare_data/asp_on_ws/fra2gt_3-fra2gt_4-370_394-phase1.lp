
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

	orientation(fra2gt_3,138,370). coords(fra2gt_3,77,121,370).  visible(fra2gt_3,370). happensAt(walking(fra2gt_3),370). orientation(fra2gt_4,155,370). coords(fra2gt_4,83,120,370).  visible(fra2gt_4,370). happensAt(walking(fra2gt_4),370).
orientation(fra2gt_3,138,371). coords(fra2gt_3,76,121,371).  visible(fra2gt_3,371). happensAt(walking(fra2gt_3),371). orientation(fra2gt_4,155,371). coords(fra2gt_4,81,123,371).  visible(fra2gt_4,371). happensAt(walking(fra2gt_4),371).
orientation(fra2gt_3,138,372). coords(fra2gt_3,76,122,372).  visible(fra2gt_3,372). happensAt(walking(fra2gt_3),372). orientation(fra2gt_4,155,372). coords(fra2gt_4,81,124,372).  visible(fra2gt_4,372). happensAt(walking(fra2gt_4),372).
orientation(fra2gt_3,138,373). coords(fra2gt_3,76,122,373).  visible(fra2gt_3,373). happensAt(walking(fra2gt_3),373). orientation(fra2gt_4,155,373). coords(fra2gt_4,81,125,373).  visible(fra2gt_4,373). happensAt(walking(fra2gt_4),373).
orientation(fra2gt_3,138,374). coords(fra2gt_3,76,122,374).  visible(fra2gt_3,374). happensAt(walking(fra2gt_3),374). orientation(fra2gt_4,155,374). coords(fra2gt_4,81,125,374).  visible(fra2gt_4,374). happensAt(walking(fra2gt_4),374).
orientation(fra2gt_3,138,375). coords(fra2gt_3,76,124,375).  visible(fra2gt_3,375). happensAt(walking(fra2gt_3),375). orientation(fra2gt_4,155,375). coords(fra2gt_4,82,126,375).  visible(fra2gt_4,375). happensAt(walking(fra2gt_4),375).
orientation(fra2gt_3,138,376). coords(fra2gt_3,76,124,376).  visible(fra2gt_3,376). happensAt(walking(fra2gt_3),376). orientation(fra2gt_4,155,376). coords(fra2gt_4,82,126,376).  visible(fra2gt_4,376). happensAt(walking(fra2gt_4),376).
orientation(fra2gt_3,138,377). coords(fra2gt_3,75,125,377).  visible(fra2gt_3,377). happensAt(walking(fra2gt_3),377). orientation(fra2gt_4,153,377). coords(fra2gt_4,80,127,377).  visible(fra2gt_4,377). happensAt(walking(fra2gt_4),377).
orientation(fra2gt_3,138,378). coords(fra2gt_3,75,125,378).  visible(fra2gt_3,378). happensAt(walking(fra2gt_3),378). orientation(fra2gt_4,163,378). coords(fra2gt_4,79,130,378).  visible(fra2gt_4,378). happensAt(walking(fra2gt_4),378).
orientation(fra2gt_3,138,379). coords(fra2gt_3,75,125,379).  visible(fra2gt_3,379). happensAt(walking(fra2gt_3),379). orientation(fra2gt_4,163,379). coords(fra2gt_4,79,130,379).  visible(fra2gt_4,379). happensAt(walking(fra2gt_4),379).
orientation(fra2gt_3,138,380). coords(fra2gt_3,75,127,380).  visible(fra2gt_3,380). happensAt(walking(fra2gt_3),380). orientation(fra2gt_4,158,380). coords(fra2gt_4,78,131,380).  visible(fra2gt_4,380). happensAt(walking(fra2gt_4),380).
orientation(fra2gt_3,138,381). coords(fra2gt_3,74,127,381).  visible(fra2gt_3,381). happensAt(walking(fra2gt_3),381). orientation(fra2gt_4,150,381). coords(fra2gt_4,77,131,381).  visible(fra2gt_4,381). happensAt(walking(fra2gt_4),381).
orientation(fra2gt_3,138,382). coords(fra2gt_3,74,129,382).  visible(fra2gt_3,382). happensAt(walking(fra2gt_3),382). orientation(fra2gt_4,149,382). coords(fra2gt_4,78,132,382).  visible(fra2gt_4,382). happensAt(walking(fra2gt_4),382).
orientation(fra2gt_3,138,383). coords(fra2gt_3,73,132,383).  visible(fra2gt_3,383). happensAt(walking(fra2gt_3),383). orientation(fra2gt_4,145,383). coords(fra2gt_4,78,131,383).  visible(fra2gt_4,383). happensAt(walking(fra2gt_4),383).
orientation(fra2gt_3,151,384). coords(fra2gt_3,71,133,384).  visible(fra2gt_3,384). happensAt(walking(fra2gt_3),384). orientation(fra2gt_4,146,384). coords(fra2gt_4,77,133,384).  visible(fra2gt_4,384). happensAt(walking(fra2gt_4),384).
orientation(fra2gt_3,159,385). coords(fra2gt_3,71,133,385).  visible(fra2gt_3,385). happensAt(walking(fra2gt_3),385). orientation(fra2gt_4,143,385). coords(fra2gt_4,77,135,385).  visible(fra2gt_4,385). happensAt(walking(fra2gt_4),385).
orientation(fra2gt_3,159,386). coords(fra2gt_3,70,133,386).  visible(fra2gt_3,386). happensAt(walking(fra2gt_3),386). orientation(fra2gt_4,142,386). coords(fra2gt_4,77,136,386).  visible(fra2gt_4,386). happensAt(walking(fra2gt_4),386).
orientation(fra2gt_3,166,387). coords(fra2gt_3,70,133,387).  visible(fra2gt_3,387). happensAt(walking(fra2gt_3),387). orientation(fra2gt_4,142,387). coords(fra2gt_4,77,137,387).  visible(fra2gt_4,387). happensAt(walking(fra2gt_4),387).
orientation(fra2gt_3,161,388). coords(fra2gt_3,69,135,388).  visible(fra2gt_3,388). happensAt(walking(fra2gt_3),388). orientation(fra2gt_4,142,388). coords(fra2gt_4,77,138,388).  visible(fra2gt_4,388). happensAt(walking(fra2gt_4),388).
orientation(fra2gt_3,161,389). coords(fra2gt_3,69,136,389).  visible(fra2gt_3,389). happensAt(walking(fra2gt_3),389). orientation(fra2gt_4,142,389). coords(fra2gt_4,77,138,389).  visible(fra2gt_4,389). happensAt(walking(fra2gt_4),389).
orientation(fra2gt_3,158,390). coords(fra2gt_3,69,137,390).  visible(fra2gt_3,390). happensAt(walking(fra2gt_3),390). orientation(fra2gt_4,142,390). coords(fra2gt_4,76,138,390).  visible(fra2gt_4,390). happensAt(walking(fra2gt_4),390).
orientation(fra2gt_3,159,391). coords(fra2gt_3,66,137,391).  visible(fra2gt_3,391). happensAt(walking(fra2gt_3),391). orientation(fra2gt_4,148,391). coords(fra2gt_4,76,138,391).  visible(fra2gt_4,391). happensAt(walking(fra2gt_4),391).
orientation(fra2gt_3,157,392). coords(fra2gt_3,66,138,392).  visible(fra2gt_3,392). happensAt(walking(fra2gt_3),392). orientation(fra2gt_4,148,392). coords(fra2gt_4,76,138,392).  visible(fra2gt_4,392). happensAt(walking(fra2gt_4),392).
orientation(fra2gt_3,167,393). coords(fra2gt_3,65,138,393).  visible(fra2gt_3,393). happensAt(walking(fra2gt_3),393). orientation(fra2gt_4,148,393). coords(fra2gt_4,77,138,393).  visible(fra2gt_4,393). happensAt(walking(fra2gt_4),393).
orientation(fra2gt_3,162,394). coords(fra2gt_3,65,138,394).  visible(fra2gt_3,394). happensAt(walking(fra2gt_3),394). orientation(fra2gt_4,148,394). coords(fra2gt_4,77,138,394).  visible(fra2gt_4,394). happensAt(walking(fra2gt_4),394).
#show holdsAt/2.