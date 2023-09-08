
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

	orientation(mwt1gt_1,160,257). coords(mwt1gt_1,73,180,257).  visible(mwt1gt_1,257). happensAt(walking(mwt1gt_1),257). orientation(mwt1gt_2,130,257). coords(mwt1gt_2,235,35,257).  visible(mwt1gt_2,257). happensAt(walking(mwt1gt_2),257).
orientation(mwt1gt_1,160,258). coords(mwt1gt_1,73,180,258).  visible(mwt1gt_1,258). happensAt(walking(mwt1gt_1),258). orientation(mwt1gt_2,130,258). coords(mwt1gt_2,233,36,258).  visible(mwt1gt_2,258). happensAt(walking(mwt1gt_2),258).
orientation(mwt1gt_1,160,259). coords(mwt1gt_1,73,180,259).  visible(mwt1gt_1,259). happensAt(walking(mwt1gt_1),259). orientation(mwt1gt_2,130,259). coords(mwt1gt_2,233,36,259).  visible(mwt1gt_2,259). happensAt(walking(mwt1gt_2),259).
orientation(mwt1gt_1,160,260). coords(mwt1gt_1,73,179,260).  visible(mwt1gt_1,260). happensAt(walking(mwt1gt_1),260). orientation(mwt1gt_2,130,260). coords(mwt1gt_2,227,36,260).  visible(mwt1gt_2,260). happensAt(walking(mwt1gt_2),260).
orientation(mwt1gt_1,160,261). coords(mwt1gt_1,73,179,261).  visible(mwt1gt_1,261). happensAt(walking(mwt1gt_1),261). orientation(mwt1gt_2,130,261). coords(mwt1gt_2,224,38,261).  visible(mwt1gt_2,261). happensAt(walking(mwt1gt_2),261).
orientation(mwt1gt_1,160,262). coords(mwt1gt_1,75,179,262).  visible(mwt1gt_1,262). happensAt(walking(mwt1gt_1),262). orientation(mwt1gt_2,130,262). coords(mwt1gt_2,222,40,262).  visible(mwt1gt_2,262). happensAt(walking(mwt1gt_2),262).
orientation(mwt1gt_1,160,263). coords(mwt1gt_1,77,179,263).  visible(mwt1gt_1,263). happensAt(walking(mwt1gt_1),263). orientation(mwt1gt_2,130,263). coords(mwt1gt_2,220,43,263).  visible(mwt1gt_2,263). happensAt(walking(mwt1gt_2),263).
orientation(mwt1gt_1,160,264). coords(mwt1gt_1,77,179,264).  visible(mwt1gt_1,264). happensAt(walking(mwt1gt_1),264). orientation(mwt1gt_2,130,264). coords(mwt1gt_2,219,44,264).  visible(mwt1gt_2,264). happensAt(walking(mwt1gt_2),264).
orientation(mwt1gt_1,160,265). coords(mwt1gt_1,78,178,265).  visible(mwt1gt_1,265). happensAt(walking(mwt1gt_1),265). orientation(mwt1gt_2,130,265). coords(mwt1gt_2,219,47,265).  visible(mwt1gt_2,265). happensAt(walking(mwt1gt_2),265).
orientation(mwt1gt_1,160,266). coords(mwt1gt_1,78,178,266).  visible(mwt1gt_1,266). happensAt(walking(mwt1gt_1),266). orientation(mwt1gt_2,130,266). coords(mwt1gt_2,218,48,266).  visible(mwt1gt_2,266). happensAt(walking(mwt1gt_2),266).
orientation(mwt1gt_1,160,267). coords(mwt1gt_1,79,178,267).  visible(mwt1gt_1,267). happensAt(walking(mwt1gt_1),267). orientation(mwt1gt_2,130,267). coords(mwt1gt_2,218,49,267).  visible(mwt1gt_2,267). happensAt(walking(mwt1gt_2),267).
orientation(mwt1gt_1,160,268). coords(mwt1gt_1,79,176,268).  visible(mwt1gt_1,268). happensAt(walking(mwt1gt_1),268). orientation(mwt1gt_2,130,268). coords(mwt1gt_2,217,51,268).  visible(mwt1gt_2,268). happensAt(walking(mwt1gt_2),268).
orientation(mwt1gt_1,160,269). coords(mwt1gt_1,79,176,269).  visible(mwt1gt_1,269). happensAt(walking(mwt1gt_1),269). orientation(mwt1gt_2,130,269). coords(mwt1gt_2,215,51,269).  visible(mwt1gt_2,269). happensAt(walking(mwt1gt_2),269).
orientation(mwt1gt_1,160,270). coords(mwt1gt_1,79,176,270).  visible(mwt1gt_1,270). happensAt(walking(mwt1gt_1),270). orientation(mwt1gt_2,130,270). coords(mwt1gt_2,213,53,270).  visible(mwt1gt_2,270). happensAt(walking(mwt1gt_2),270).
orientation(mwt1gt_1,160,271). coords(mwt1gt_1,79,176,271).  visible(mwt1gt_1,271). happensAt(walking(mwt1gt_1),271). orientation(mwt1gt_2,130,271). coords(mwt1gt_2,211,54,271).  visible(mwt1gt_2,271). happensAt(walking(mwt1gt_2),271).
orientation(mwt1gt_1,160,272). coords(mwt1gt_1,79,176,272).  visible(mwt1gt_1,272). happensAt(walking(mwt1gt_1),272). orientation(mwt1gt_2,130,272). coords(mwt1gt_2,208,54,272).  visible(mwt1gt_2,272). happensAt(walking(mwt1gt_2),272).
orientation(mwt1gt_1,160,273). coords(mwt1gt_1,79,175,273).  visible(mwt1gt_1,273). happensAt(walking(mwt1gt_1),273). orientation(mwt1gt_2,130,273). coords(mwt1gt_2,204,54,273).  visible(mwt1gt_2,273). happensAt(walking(mwt1gt_2),273).
orientation(mwt1gt_1,160,274). coords(mwt1gt_1,81,175,274).  visible(mwt1gt_1,274). happensAt(walking(mwt1gt_1),274). orientation(mwt1gt_2,130,274). coords(mwt1gt_2,200,55,274).  visible(mwt1gt_2,274). happensAt(walking(mwt1gt_2),274).
orientation(mwt1gt_1,160,275). coords(mwt1gt_1,81,175,275).  visible(mwt1gt_1,275). happensAt(walking(mwt1gt_1),275). orientation(mwt1gt_2,130,275). coords(mwt1gt_2,199,56,275).  visible(mwt1gt_2,275). happensAt(walking(mwt1gt_2),275).
orientation(mwt1gt_1,160,276). coords(mwt1gt_1,83,175,276).  visible(mwt1gt_1,276). happensAt(walking(mwt1gt_1),276). orientation(mwt1gt_2,130,276). coords(mwt1gt_2,198,57,276).  visible(mwt1gt_2,276). happensAt(walking(mwt1gt_2),276).
orientation(mwt1gt_1,160,277). coords(mwt1gt_1,84,175,277).  visible(mwt1gt_1,277). happensAt(walking(mwt1gt_1),277). orientation(mwt1gt_2,130,277). coords(mwt1gt_2,197,60,277).  visible(mwt1gt_2,277). happensAt(walking(mwt1gt_2),277).
orientation(mwt1gt_1,160,278). coords(mwt1gt_1,84,175,278).  visible(mwt1gt_1,278). happensAt(walking(mwt1gt_1),278). orientation(mwt1gt_2,130,278). coords(mwt1gt_2,196,61,278).  visible(mwt1gt_2,278). happensAt(walking(mwt1gt_2),278).
orientation(mwt1gt_1,160,279). coords(mwt1gt_1,86,175,279).  visible(mwt1gt_1,279). happensAt(walking(mwt1gt_1),279). orientation(mwt1gt_2,130,279). coords(mwt1gt_2,195,64,279).  visible(mwt1gt_2,279). happensAt(walking(mwt1gt_2),279).
orientation(mwt1gt_1,160,280). coords(mwt1gt_1,87,175,280).  visible(mwt1gt_1,280). happensAt(walking(mwt1gt_1),280). orientation(mwt1gt_2,130,280). coords(mwt1gt_2,194,65,280).  visible(mwt1gt_2,280). happensAt(walking(mwt1gt_2),280).
orientation(mwt1gt_1,160,281). coords(mwt1gt_1,88,174,281).  visible(mwt1gt_1,281). happensAt(walking(mwt1gt_1),281). orientation(mwt1gt_2,130,281). coords(mwt1gt_2,193,67,281).  visible(mwt1gt_2,281). happensAt(walking(mwt1gt_2),281).
#show holdsAt/2.