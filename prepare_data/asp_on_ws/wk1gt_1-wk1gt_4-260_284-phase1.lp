
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

	orientation(wk1gt_1,11,260). coords(wk1gt_1,224,264,260).  visible(wk1gt_1,260). happensAt(walking(wk1gt_1),260). orientation(wk1gt_4,160,260). coords(wk1gt_4,48,180,260).  visible(wk1gt_4,260). happensAt(active(wk1gt_4),260).
orientation(wk1gt_1,17,261). coords(wk1gt_1,224,260,261).  visible(wk1gt_1,261). happensAt(walking(wk1gt_1),261). orientation(wk1gt_4,159,261). coords(wk1gt_4,48,179,261).  visible(wk1gt_4,261). happensAt(active(wk1gt_4),261).
orientation(wk1gt_1,22,262). coords(wk1gt_1,224,256,262).  visible(wk1gt_1,262). happensAt(walking(wk1gt_1),262). orientation(wk1gt_4,160,262). coords(wk1gt_4,48,179,262).  visible(wk1gt_4,262). happensAt(active(wk1gt_4),262).
orientation(wk1gt_1,22,263). coords(wk1gt_1,223,254,263).  visible(wk1gt_1,263). happensAt(walking(wk1gt_1),263). orientation(wk1gt_4,158,263). coords(wk1gt_4,48,179,263).  visible(wk1gt_4,263). happensAt(active(wk1gt_4),263).
orientation(wk1gt_1,22,264). coords(wk1gt_1,223,252,264).  visible(wk1gt_1,264). happensAt(walking(wk1gt_1),264). orientation(wk1gt_4,162,264). coords(wk1gt_4,48,179,264).  visible(wk1gt_4,264). happensAt(active(wk1gt_4),264).
orientation(wk1gt_1,22,265). coords(wk1gt_1,222,251,265).  visible(wk1gt_1,265). happensAt(walking(wk1gt_1),265). orientation(wk1gt_4,163,265). coords(wk1gt_4,48,179,265).  visible(wk1gt_4,265). happensAt(active(wk1gt_4),265).
orientation(wk1gt_1,22,266). coords(wk1gt_1,222,251,266).  visible(wk1gt_1,266). happensAt(walking(wk1gt_1),266). orientation(wk1gt_4,154,266). coords(wk1gt_4,48,179,266).  visible(wk1gt_4,266). happensAt(active(wk1gt_4),266).
orientation(wk1gt_1,8,267). coords(wk1gt_1,221,250,267).  visible(wk1gt_1,267). happensAt(walking(wk1gt_1),267). orientation(wk1gt_4,145,267). coords(wk1gt_4,48,179,267).  visible(wk1gt_4,267). happensAt(active(wk1gt_4),267).
orientation(wk1gt_1,8,268). coords(wk1gt_1,218,249,268).  visible(wk1gt_1,268). happensAt(walking(wk1gt_1),268). orientation(wk1gt_4,152,268). coords(wk1gt_4,48,179,268).  visible(wk1gt_4,268). happensAt(active(wk1gt_4),268).
orientation(wk1gt_1,8,269). coords(wk1gt_1,218,247,269).  visible(wk1gt_1,269). happensAt(walking(wk1gt_1),269). orientation(wk1gt_4,156,269). coords(wk1gt_4,48,178,269).  visible(wk1gt_4,269). happensAt(active(wk1gt_4),269).
orientation(wk1gt_1,3,270). coords(wk1gt_1,216,245,270).  visible(wk1gt_1,270). happensAt(walking(wk1gt_1),270). orientation(wk1gt_4,158,270). coords(wk1gt_4,48,178,270).  visible(wk1gt_4,270). happensAt(active(wk1gt_4),270).
orientation(wk1gt_1,3,271). coords(wk1gt_1,216,242,271).  visible(wk1gt_1,271). happensAt(walking(wk1gt_1),271). orientation(wk1gt_4,157,271). coords(wk1gt_4,49,178,271).  visible(wk1gt_4,271). happensAt(active(wk1gt_4),271).
orientation(wk1gt_1,3,272). coords(wk1gt_1,213,241,272).  visible(wk1gt_1,272). happensAt(walking(wk1gt_1),272). orientation(wk1gt_4,153,272). coords(wk1gt_4,49,178,272).  visible(wk1gt_4,272). happensAt(active(wk1gt_4),272).
orientation(wk1gt_1,3,273). coords(wk1gt_1,212,239,273).  visible(wk1gt_1,273). happensAt(walking(wk1gt_1),273). orientation(wk1gt_4,157,273). coords(wk1gt_4,49,178,273).  visible(wk1gt_4,273). happensAt(active(wk1gt_4),273).
orientation(wk1gt_1,3,274). coords(wk1gt_1,212,236,274).  visible(wk1gt_1,274). happensAt(walking(wk1gt_1),274). orientation(wk1gt_4,156,274). coords(wk1gt_4,49,178,274).  visible(wk1gt_4,274). happensAt(active(wk1gt_4),274).
orientation(wk1gt_1,3,275). coords(wk1gt_1,211,232,275).  visible(wk1gt_1,275). happensAt(walking(wk1gt_1),275). orientation(wk1gt_4,153,275). coords(wk1gt_4,49,178,275).  visible(wk1gt_4,275). happensAt(active(wk1gt_4),275).
orientation(wk1gt_1,9,276). coords(wk1gt_1,211,230,276).  visible(wk1gt_1,276). happensAt(walking(wk1gt_1),276). orientation(wk1gt_4,154,276). coords(wk1gt_4,49,177,276).  visible(wk1gt_4,276). happensAt(active(wk1gt_4),276).
orientation(wk1gt_1,9,277). coords(wk1gt_1,210,229,277).  visible(wk1gt_1,277). happensAt(walking(wk1gt_1),277). orientation(wk1gt_4,154,277). coords(wk1gt_4,49,177,277).  visible(wk1gt_4,277). happensAt(active(wk1gt_4),277).
orientation(wk1gt_1,174,278). coords(wk1gt_1,210,229,278).  visible(wk1gt_1,278). happensAt(walking(wk1gt_1),278). orientation(wk1gt_4,156,278). coords(wk1gt_4,50,177,278).  visible(wk1gt_4,278). happensAt(active(wk1gt_4),278).
orientation(wk1gt_1,167,279). coords(wk1gt_1,210,229,279).  visible(wk1gt_1,279). happensAt(walking(wk1gt_1),279). orientation(wk1gt_4,155,279). coords(wk1gt_4,51,177,279).  visible(wk1gt_4,279). happensAt(active(wk1gt_4),279).
orientation(wk1gt_1,167,280). coords(wk1gt_1,209,228,280).  visible(wk1gt_1,280). happensAt(walking(wk1gt_1),280). orientation(wk1gt_4,156,280). coords(wk1gt_4,51,177,280).  visible(wk1gt_4,280). happensAt(active(wk1gt_4),280).
orientation(wk1gt_1,155,281). coords(wk1gt_1,207,227,281).  visible(wk1gt_1,281). happensAt(walking(wk1gt_1),281). orientation(wk1gt_4,157,281). coords(wk1gt_4,51,176,281).  visible(wk1gt_4,281). happensAt(active(wk1gt_4),281).
orientation(wk1gt_1,155,282). coords(wk1gt_1,206,224,282).  visible(wk1gt_1,282). happensAt(walking(wk1gt_1),282). orientation(wk1gt_4,161,282). coords(wk1gt_4,51,176,282).  visible(wk1gt_4,282). happensAt(active(wk1gt_4),282).
orientation(wk1gt_1,155,283). coords(wk1gt_1,203,222,283).  visible(wk1gt_1,283). happensAt(walking(wk1gt_1),283). orientation(wk1gt_4,162,283). coords(wk1gt_4,51,176,283).  visible(wk1gt_4,283). happensAt(active(wk1gt_4),283).
orientation(wk1gt_1,155,284). coords(wk1gt_1,203,218,284).  visible(wk1gt_1,284). happensAt(walking(wk1gt_1),284). orientation(wk1gt_4,159,284). coords(wk1gt_4,51,176,284).  visible(wk1gt_4,284). happensAt(active(wk1gt_4),284).
#show holdsAt/2.