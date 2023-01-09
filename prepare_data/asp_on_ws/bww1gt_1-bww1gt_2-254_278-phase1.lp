
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

	orientation(bww1gt_1,176,254). coords(bww1gt_1,85,254,254).  visible(bww1gt_1,254). happensAt(walking(bww1gt_1),254). orientation(bww1gt_2,167,254). coords(bww1gt_2,42,165,254).  visible(bww1gt_2,254). happensAt(walking(bww1gt_2),254).
orientation(bww1gt_1,176,255). coords(bww1gt_1,85,253,255).  visible(bww1gt_1,255). happensAt(walking(bww1gt_1),255). orientation(bww1gt_2,167,255). coords(bww1gt_2,42,165,255).  visible(bww1gt_2,255). happensAt(walking(bww1gt_2),255).
orientation(bww1gt_1,1,256). coords(bww1gt_1,85,253,256).  visible(bww1gt_1,256). happensAt(walking(bww1gt_1),256). orientation(bww1gt_2,167,256). coords(bww1gt_2,43,165,256).  visible(bww1gt_2,256). happensAt(walking(bww1gt_2),256).
orientation(bww1gt_1,178,257). coords(bww1gt_1,86,252,257).  visible(bww1gt_1,257). happensAt(walking(bww1gt_1),257). orientation(bww1gt_2,167,257). coords(bww1gt_2,44,165,257).  visible(bww1gt_2,257). happensAt(walking(bww1gt_2),257).
orientation(bww1gt_1,171,258). coords(bww1gt_1,86,252,258).  visible(bww1gt_1,258). happensAt(walking(bww1gt_1),258). orientation(bww1gt_2,167,258). coords(bww1gt_2,45,165,258).  visible(bww1gt_2,258). happensAt(walking(bww1gt_2),258).
orientation(bww1gt_1,176,259). coords(bww1gt_1,87,252,259).  visible(bww1gt_1,259). happensAt(walking(bww1gt_1),259). orientation(bww1gt_2,167,259). coords(bww1gt_2,46,165,259).  visible(bww1gt_2,259). happensAt(walking(bww1gt_2),259).
orientation(bww1gt_1,172,260). coords(bww1gt_1,87,251,260).  visible(bww1gt_1,260). happensAt(walking(bww1gt_1),260). orientation(bww1gt_2,167,260). coords(bww1gt_2,46,165,260).  visible(bww1gt_2,260). happensAt(walking(bww1gt_2),260).
orientation(bww1gt_1,173,261). coords(bww1gt_1,87,250,261).  visible(bww1gt_1,261). happensAt(walking(bww1gt_1),261). orientation(bww1gt_2,167,261). coords(bww1gt_2,46,165,261).  visible(bww1gt_2,261). happensAt(walking(bww1gt_2),261).
orientation(bww1gt_1,164,262). coords(bww1gt_1,88,250,262).  visible(bww1gt_1,262). happensAt(walking(bww1gt_1),262). orientation(bww1gt_2,167,262). coords(bww1gt_2,46,165,262).  visible(bww1gt_2,262). happensAt(walking(bww1gt_2),262).
orientation(bww1gt_1,170,263). coords(bww1gt_1,88,249,263).  visible(bww1gt_1,263). happensAt(walking(bww1gt_1),263). orientation(bww1gt_2,167,263). coords(bww1gt_2,47,165,263).  visible(bww1gt_2,263). happensAt(walking(bww1gt_2),263).
orientation(bww1gt_1,169,264). coords(bww1gt_1,88,248,264).  visible(bww1gt_1,264). happensAt(walking(bww1gt_1),264). orientation(bww1gt_2,167,264). coords(bww1gt_2,47,165,264).  visible(bww1gt_2,264). happensAt(walking(bww1gt_2),264).
orientation(bww1gt_1,165,265). coords(bww1gt_1,89,248,265).  visible(bww1gt_1,265). happensAt(walking(bww1gt_1),265). orientation(bww1gt_2,167,265). coords(bww1gt_2,47,165,265).  visible(bww1gt_2,265). happensAt(walking(bww1gt_2),265).
orientation(bww1gt_1,168,266). coords(bww1gt_1,89,248,266).  visible(bww1gt_1,266). happensAt(walking(bww1gt_1),266). orientation(bww1gt_2,167,266). coords(bww1gt_2,47,165,266).  visible(bww1gt_2,266). happensAt(walking(bww1gt_2),266).
orientation(bww1gt_1,164,267). coords(bww1gt_1,89,248,267).  visible(bww1gt_1,267). happensAt(walking(bww1gt_1),267). orientation(bww1gt_2,167,267). coords(bww1gt_2,47,165,267).  visible(bww1gt_2,267). happensAt(walking(bww1gt_2),267).
orientation(bww1gt_1,164,268). coords(bww1gt_1,90,248,268).  visible(bww1gt_1,268). happensAt(walking(bww1gt_1),268). orientation(bww1gt_2,167,268). coords(bww1gt_2,47,165,268).  visible(bww1gt_2,268). happensAt(walking(bww1gt_2),268).
orientation(bww1gt_1,159,269). coords(bww1gt_1,90,249,269).  visible(bww1gt_1,269). happensAt(walking(bww1gt_1),269). orientation(bww1gt_2,167,269). coords(bww1gt_2,47,165,269).  visible(bww1gt_2,269). happensAt(walking(bww1gt_2),269).
orientation(bww1gt_1,157,270). coords(bww1gt_1,92,249,270).  visible(bww1gt_1,270). happensAt(walking(bww1gt_1),270). orientation(bww1gt_2,167,270). coords(bww1gt_2,47,165,270).  visible(bww1gt_2,270). happensAt(walking(bww1gt_2),270).
orientation(bww1gt_1,157,271). coords(bww1gt_1,92,249,271).  visible(bww1gt_1,271). happensAt(walking(bww1gt_1),271). orientation(bww1gt_2,167,271). coords(bww1gt_2,47,165,271).  visible(bww1gt_2,271). happensAt(walking(bww1gt_2),271).
orientation(bww1gt_1,156,272). coords(bww1gt_1,93,250,272).  visible(bww1gt_1,272). happensAt(walking(bww1gt_1),272). orientation(bww1gt_2,167,272). coords(bww1gt_2,47,165,272).  visible(bww1gt_2,272). happensAt(walking(bww1gt_2),272).
orientation(bww1gt_1,160,273). coords(bww1gt_1,93,251,273).  visible(bww1gt_1,273). happensAt(walking(bww1gt_1),273). orientation(bww1gt_2,167,273). coords(bww1gt_2,48,165,273).  visible(bww1gt_2,273). happensAt(walking(bww1gt_2),273).
orientation(bww1gt_1,158,274). coords(bww1gt_1,93,251,274).  visible(bww1gt_1,274). happensAt(walking(bww1gt_1),274). orientation(bww1gt_2,167,274). coords(bww1gt_2,49,165,274).  visible(bww1gt_2,274). happensAt(walking(bww1gt_2),274).
orientation(bww1gt_1,168,275). coords(bww1gt_1,94,252,275).  visible(bww1gt_1,275). happensAt(walking(bww1gt_1),275). orientation(bww1gt_2,167,275). coords(bww1gt_2,49,165,275).  visible(bww1gt_2,275). happensAt(walking(bww1gt_2),275).
orientation(bww1gt_1,167,276). coords(bww1gt_1,94,253,276).  visible(bww1gt_1,276). happensAt(walking(bww1gt_1),276). orientation(bww1gt_2,167,276). coords(bww1gt_2,49,165,276).  visible(bww1gt_2,276). happensAt(walking(bww1gt_2),276).
orientation(bww1gt_1,165,277). coords(bww1gt_1,94,254,277).  visible(bww1gt_1,277). happensAt(walking(bww1gt_1),277). orientation(bww1gt_2,167,277). coords(bww1gt_2,50,165,277).  visible(bww1gt_2,277). happensAt(walking(bww1gt_2),277).
orientation(bww1gt_1,166,278). coords(bww1gt_1,95,256,278).  visible(bww1gt_1,278). happensAt(walking(bww1gt_1),278). orientation(bww1gt_2,167,278). coords(bww1gt_2,50,165,278).  visible(bww1gt_2,278). happensAt(walking(bww1gt_2),278).
#show holdsAt/2.