
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

	orientation(ms3ggt_0,158,368). coords(ms3ggt_0,134,142,368).  visible(ms3ggt_0,368). happensAt(walking(ms3ggt_0),368). orientation(ms3ggt_2,173,368). coords(ms3ggt_2,180,218,368).  visible(ms3ggt_2,368). happensAt(walking(ms3ggt_2),368).
orientation(ms3ggt_0,158,369). coords(ms3ggt_0,134,142,369).  visible(ms3ggt_0,369). happensAt(walking(ms3ggt_0),369). orientation(ms3ggt_2,173,369). coords(ms3ggt_2,178,217,369).  visible(ms3ggt_2,369). happensAt(walking(ms3ggt_2),369).
orientation(ms3ggt_0,158,370). coords(ms3ggt_0,135,142,370).  visible(ms3ggt_0,370). happensAt(walking(ms3ggt_0),370). orientation(ms3ggt_2,173,370). coords(ms3ggt_2,175,217,370).  visible(ms3ggt_2,370). happensAt(walking(ms3ggt_2),370).
orientation(ms3ggt_0,158,371). coords(ms3ggt_0,136,142,371).  visible(ms3ggt_0,371). happensAt(walking(ms3ggt_0),371). orientation(ms3ggt_2,173,371). coords(ms3ggt_2,174,217,371).  visible(ms3ggt_2,371). happensAt(walking(ms3ggt_2),371).
orientation(ms3ggt_0,158,372). coords(ms3ggt_0,136,142,372).  visible(ms3ggt_0,372). happensAt(walking(ms3ggt_0),372). orientation(ms3ggt_2,173,372). coords(ms3ggt_2,171,216,372).  visible(ms3ggt_2,372). happensAt(walking(ms3ggt_2),372).
orientation(ms3ggt_0,158,373). coords(ms3ggt_0,137,142,373).  visible(ms3ggt_0,373). happensAt(walking(ms3ggt_0),373). orientation(ms3ggt_2,173,373). coords(ms3ggt_2,170,216,373).  visible(ms3ggt_2,373). happensAt(walking(ms3ggt_2),373).
orientation(ms3ggt_0,158,374). coords(ms3ggt_0,137,142,374).  visible(ms3ggt_0,374). happensAt(walking(ms3ggt_0),374). orientation(ms3ggt_2,173,374). coords(ms3ggt_2,169,215,374).  visible(ms3ggt_2,374). happensAt(walking(ms3ggt_2),374).
orientation(ms3ggt_0,158,375). coords(ms3ggt_0,137,142,375).  visible(ms3ggt_0,375). happensAt(walking(ms3ggt_0),375). orientation(ms3ggt_2,173,375). coords(ms3ggt_2,169,215,375).  visible(ms3ggt_2,375). happensAt(walking(ms3ggt_2),375).
orientation(ms3ggt_0,158,376). coords(ms3ggt_0,137,142,376).  visible(ms3ggt_0,376). happensAt(walking(ms3ggt_0),376). orientation(ms3ggt_2,173,376). coords(ms3ggt_2,168,212,376).  visible(ms3ggt_2,376). happensAt(walking(ms3ggt_2),376).
orientation(ms3ggt_0,158,377). coords(ms3ggt_0,139,142,377).  visible(ms3ggt_0,377). happensAt(active(ms3ggt_0),377). orientation(ms3ggt_2,173,377). coords(ms3ggt_2,168,212,377).  visible(ms3ggt_2,377). happensAt(walking(ms3ggt_2),377).
orientation(ms3ggt_0,158,378). coords(ms3ggt_0,140,142,378).  visible(ms3ggt_0,378). happensAt(active(ms3ggt_0),378). orientation(ms3ggt_2,173,378). coords(ms3ggt_2,167,211,378).  visible(ms3ggt_2,378). happensAt(walking(ms3ggt_2),378).
orientation(ms3ggt_0,158,379). coords(ms3ggt_0,140,142,379).  visible(ms3ggt_0,379). happensAt(active(ms3ggt_0),379). orientation(ms3ggt_2,173,379). coords(ms3ggt_2,166,210,379).  visible(ms3ggt_2,379). happensAt(walking(ms3ggt_2),379).
orientation(ms3ggt_0,158,380). coords(ms3ggt_0,141,142,380).  visible(ms3ggt_0,380). happensAt(active(ms3ggt_0),380). orientation(ms3ggt_2,173,380). coords(ms3ggt_2,166,210,380).  visible(ms3ggt_2,380). happensAt(walking(ms3ggt_2),380).
orientation(ms3ggt_0,152,381). coords(ms3ggt_0,141,142,381).  visible(ms3ggt_0,381). happensAt(active(ms3ggt_0),381). orientation(ms3ggt_2,173,381). coords(ms3ggt_2,165,209,381).  visible(ms3ggt_2,381). happensAt(walking(ms3ggt_2),381).
orientation(ms3ggt_0,152,382). coords(ms3ggt_0,142,141,382).  visible(ms3ggt_0,382). happensAt(active(ms3ggt_0),382). orientation(ms3ggt_2,165,382). coords(ms3ggt_2,163,208,382).  visible(ms3ggt_2,382). happensAt(walking(ms3ggt_2),382).
orientation(ms3ggt_0,152,383). coords(ms3ggt_0,142,141,383).  visible(ms3ggt_0,383). happensAt(active(ms3ggt_0),383). orientation(ms3ggt_2,165,383). coords(ms3ggt_2,159,206,383).  visible(ms3ggt_2,383). happensAt(walking(ms3ggt_2),383).
orientation(ms3ggt_0,152,384). coords(ms3ggt_0,142,140,384).  visible(ms3ggt_0,384). happensAt(active(ms3ggt_0),384). orientation(ms3ggt_2,165,384). coords(ms3ggt_2,159,204,384).  visible(ms3ggt_2,384). happensAt(walking(ms3ggt_2),384).
orientation(ms3ggt_0,152,385). coords(ms3ggt_0,142,140,385).  visible(ms3ggt_0,385). happensAt(active(ms3ggt_0),385). orientation(ms3ggt_2,168,385). coords(ms3ggt_2,155,203,385).  visible(ms3ggt_2,385). happensAt(walking(ms3ggt_2),385).
orientation(ms3ggt_0,152,386). coords(ms3ggt_0,142,140,386).  visible(ms3ggt_0,386). happensAt(active(ms3ggt_0),386). orientation(ms3ggt_2,168,386). coords(ms3ggt_2,154,203,386).  visible(ms3ggt_2,386). happensAt(walking(ms3ggt_2),386).
orientation(ms3ggt_0,152,387). coords(ms3ggt_0,142,140,387).  visible(ms3ggt_0,387). happensAt(active(ms3ggt_0),387). orientation(ms3ggt_2,168,387). coords(ms3ggt_2,153,201,387).  visible(ms3ggt_2,387). happensAt(walking(ms3ggt_2),387).
orientation(ms3ggt_0,152,388). coords(ms3ggt_0,142,140,388).  visible(ms3ggt_0,388). happensAt(active(ms3ggt_0),388). orientation(ms3ggt_2,168,388). coords(ms3ggt_2,153,201,388).  visible(ms3ggt_2,388). happensAt(walking(ms3ggt_2),388).
orientation(ms3ggt_0,152,389). coords(ms3ggt_0,142,139,389).  visible(ms3ggt_0,389). happensAt(active(ms3ggt_0),389). orientation(ms3ggt_2,168,389). coords(ms3ggt_2,153,201,389).  visible(ms3ggt_2,389). happensAt(walking(ms3ggt_2),389).
orientation(ms3ggt_0,152,390). coords(ms3ggt_0,142,139,390).  visible(ms3ggt_0,390). happensAt(active(ms3ggt_0),390). orientation(ms3ggt_2,168,390). coords(ms3ggt_2,152,201,390).  visible(ms3ggt_2,390). happensAt(walking(ms3ggt_2),390).
orientation(ms3ggt_0,152,391). coords(ms3ggt_0,142,138,391).  visible(ms3ggt_0,391). happensAt(active(ms3ggt_0),391). orientation(ms3ggt_2,168,391). coords(ms3ggt_2,152,200,391).  visible(ms3ggt_2,391). happensAt(walking(ms3ggt_2),391).
orientation(ms3ggt_0,152,392). coords(ms3ggt_0,142,138,392).  visible(ms3ggt_0,392). happensAt(active(ms3ggt_0),392). orientation(ms3ggt_2,168,392). coords(ms3ggt_2,151,200,392).  visible(ms3ggt_2,392). happensAt(walking(ms3ggt_2),392).
#show holdsAt/2.