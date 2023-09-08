
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

	orientation(fra2gt_3,151,262). coords(fra2gt_3,67,136,262).  visible(fra2gt_3,262). happensAt(walking(fra2gt_3),262). orientation(fra2gt_4,155,262). coords(fra2gt_4,148,146,262).  visible(fra2gt_4,262). happensAt(walking(fra2gt_4),262).
orientation(fra2gt_3,151,263). coords(fra2gt_3,68,135,263).  visible(fra2gt_3,263). happensAt(walking(fra2gt_3),263). orientation(fra2gt_4,155,263). coords(fra2gt_4,146,145,263).  visible(fra2gt_4,263). happensAt(walking(fra2gt_4),263).
orientation(fra2gt_3,151,264). coords(fra2gt_3,69,134,264).  visible(fra2gt_3,264). happensAt(walking(fra2gt_3),264). orientation(fra2gt_4,155,264). coords(fra2gt_4,145,144,264).  visible(fra2gt_4,264). happensAt(walking(fra2gt_4),264).
orientation(fra2gt_3,151,265). coords(fra2gt_3,71,134,265).  visible(fra2gt_3,265). happensAt(walking(fra2gt_3),265). orientation(fra2gt_4,155,265). coords(fra2gt_4,145,144,265).  visible(fra2gt_4,265). happensAt(walking(fra2gt_4),265).
orientation(fra2gt_3,151,266). coords(fra2gt_3,72,134,266).  visible(fra2gt_3,266). happensAt(walking(fra2gt_3),266). orientation(fra2gt_4,155,266). coords(fra2gt_4,145,144,266).  visible(fra2gt_4,266). happensAt(walking(fra2gt_4),266).
orientation(fra2gt_3,156,267). coords(fra2gt_3,73,133,267).  visible(fra2gt_3,267). happensAt(walking(fra2gt_3),267). orientation(fra2gt_4,155,267). coords(fra2gt_4,143,143,267).  visible(fra2gt_4,267). happensAt(walking(fra2gt_4),267).
orientation(fra2gt_3,156,268). coords(fra2gt_3,73,133,268).  visible(fra2gt_3,268). happensAt(walking(fra2gt_3),268). orientation(fra2gt_4,155,268). coords(fra2gt_4,142,142,268).  visible(fra2gt_4,268). happensAt(walking(fra2gt_4),268).
orientation(fra2gt_3,156,269). coords(fra2gt_3,73,132,269).  visible(fra2gt_3,269). happensAt(walking(fra2gt_3),269). orientation(fra2gt_4,155,269). coords(fra2gt_4,141,141,269).  visible(fra2gt_4,269). happensAt(walking(fra2gt_4),269).
orientation(fra2gt_3,156,270). coords(fra2gt_3,74,132,270).  visible(fra2gt_3,270). happensAt(walking(fra2gt_3),270). orientation(fra2gt_4,155,270). coords(fra2gt_4,139,141,270).  visible(fra2gt_4,270). happensAt(walking(fra2gt_4),270).
orientation(fra2gt_3,156,271). coords(fra2gt_3,74,131,271).  visible(fra2gt_3,271). happensAt(walking(fra2gt_3),271). orientation(fra2gt_4,155,271). coords(fra2gt_4,137,140,271).  visible(fra2gt_4,271). happensAt(walking(fra2gt_4),271).
orientation(fra2gt_3,156,272). coords(fra2gt_3,75,130,272).  visible(fra2gt_3,272). happensAt(walking(fra2gt_3),272). orientation(fra2gt_4,155,272). coords(fra2gt_4,136,139,272).  visible(fra2gt_4,272). happensAt(walking(fra2gt_4),272).
orientation(fra2gt_3,156,273). coords(fra2gt_3,75,130,273).  visible(fra2gt_3,273). happensAt(walking(fra2gt_3),273). orientation(fra2gt_4,155,273). coords(fra2gt_4,135,137,273).  visible(fra2gt_4,273). happensAt(walking(fra2gt_4),273).
orientation(fra2gt_3,156,274). coords(fra2gt_3,75,129,274).  visible(fra2gt_3,274). happensAt(walking(fra2gt_3),274). orientation(fra2gt_4,155,274). coords(fra2gt_4,135,136,274).  visible(fra2gt_4,274). happensAt(walking(fra2gt_4),274).
orientation(fra2gt_3,156,275). coords(fra2gt_3,75,129,275).  visible(fra2gt_3,275). happensAt(walking(fra2gt_3),275). orientation(fra2gt_4,155,275). coords(fra2gt_4,134,132,275).  visible(fra2gt_4,275). happensAt(walking(fra2gt_4),275).
orientation(fra2gt_3,156,276). coords(fra2gt_3,76,128,276).  visible(fra2gt_3,276). happensAt(walking(fra2gt_3),276). orientation(fra2gt_4,155,276). coords(fra2gt_4,133,130,276).  visible(fra2gt_4,276). happensAt(walking(fra2gt_4),276).
orientation(fra2gt_3,156,277). coords(fra2gt_3,77,128,277).  visible(fra2gt_3,277). happensAt(walking(fra2gt_3),277). orientation(fra2gt_4,155,277). coords(fra2gt_4,132,129,277).  visible(fra2gt_4,277). happensAt(walking(fra2gt_4),277).
orientation(fra2gt_3,156,278). coords(fra2gt_3,79,128,278).  visible(fra2gt_3,278). happensAt(walking(fra2gt_3),278). orientation(fra2gt_4,155,278). coords(fra2gt_4,131,129,278).  visible(fra2gt_4,278). happensAt(walking(fra2gt_4),278).
orientation(fra2gt_3,156,279). coords(fra2gt_3,80,128,279).  visible(fra2gt_3,279). happensAt(walking(fra2gt_3),279). orientation(fra2gt_4,155,279). coords(fra2gt_4,130,127,279).  visible(fra2gt_4,279). happensAt(walking(fra2gt_4),279).
orientation(fra2gt_3,156,280). coords(fra2gt_3,80,127,280).  visible(fra2gt_3,280). happensAt(walking(fra2gt_3),280). orientation(fra2gt_4,155,280). coords(fra2gt_4,128,128,280).  visible(fra2gt_4,280). happensAt(walking(fra2gt_4),280).
orientation(fra2gt_3,156,281). coords(fra2gt_3,80,127,281).  visible(fra2gt_3,281). happensAt(walking(fra2gt_3),281). orientation(fra2gt_4,155,281). coords(fra2gt_4,125,127,281).  visible(fra2gt_4,281). happensAt(walking(fra2gt_4),281).
orientation(fra2gt_3,156,282). coords(fra2gt_3,80,126,282).  visible(fra2gt_3,282). happensAt(walking(fra2gt_3),282). orientation(fra2gt_4,155,282). coords(fra2gt_4,123,126,282).  visible(fra2gt_4,282). happensAt(walking(fra2gt_4),282).
orientation(fra2gt_3,156,283). coords(fra2gt_3,81,126,283).  visible(fra2gt_3,283). happensAt(walking(fra2gt_3),283). orientation(fra2gt_4,155,283). coords(fra2gt_4,122,126,283).  visible(fra2gt_4,283). happensAt(walking(fra2gt_4),283).
orientation(fra2gt_3,153,284). coords(fra2gt_3,81,126,284).  visible(fra2gt_3,284). happensAt(walking(fra2gt_3),284). orientation(fra2gt_4,155,284). coords(fra2gt_4,121,127,284).  visible(fra2gt_4,284). happensAt(walking(fra2gt_4),284).
orientation(fra2gt_3,153,285). coords(fra2gt_3,81,124,285).  visible(fra2gt_3,285). happensAt(walking(fra2gt_3),285). orientation(fra2gt_4,155,285). coords(fra2gt_4,121,128,285).  visible(fra2gt_4,285). happensAt(walking(fra2gt_4),285).
orientation(fra2gt_3,153,286). coords(fra2gt_3,81,124,286).  visible(fra2gt_3,286). happensAt(walking(fra2gt_3),286). orientation(fra2gt_4,155,286). coords(fra2gt_4,120,129,286).  visible(fra2gt_4,286). happensAt(walking(fra2gt_4),286).
#show holdsAt/2.