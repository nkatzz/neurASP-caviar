
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

	orientation(wk1gt_4,163,255). coords(wk1gt_4,47,181,255).  visible(wk1gt_4,255). happensAt(active(wk1gt_4),255). orientation(wk1gt_5,162,255). coords(wk1gt_5,50,166,255).  visible(wk1gt_5,255). happensAt(inactive(wk1gt_5),255).
orientation(wk1gt_4,161,256). coords(wk1gt_4,47,181,256).  visible(wk1gt_4,256). happensAt(active(wk1gt_4),256). orientation(wk1gt_5,162,256). coords(wk1gt_5,50,166,256).  visible(wk1gt_5,256). happensAt(inactive(wk1gt_5),256).
orientation(wk1gt_4,160,257). coords(wk1gt_4,47,181,257).  visible(wk1gt_4,257). happensAt(active(wk1gt_4),257). orientation(wk1gt_5,162,257). coords(wk1gt_5,50,166,257).  visible(wk1gt_5,257). happensAt(inactive(wk1gt_5),257).
orientation(wk1gt_4,157,258). coords(wk1gt_4,47,181,258).  visible(wk1gt_4,258). happensAt(active(wk1gt_4),258). orientation(wk1gt_5,162,258). coords(wk1gt_5,50,166,258).  visible(wk1gt_5,258). happensAt(inactive(wk1gt_5),258).
orientation(wk1gt_4,160,259). coords(wk1gt_4,47,180,259).  visible(wk1gt_4,259). happensAt(active(wk1gt_4),259). orientation(wk1gt_5,162,259). coords(wk1gt_5,50,166,259).  visible(wk1gt_5,259). happensAt(inactive(wk1gt_5),259).
orientation(wk1gt_4,160,260). coords(wk1gt_4,48,180,260).  visible(wk1gt_4,260). happensAt(active(wk1gt_4),260). orientation(wk1gt_5,162,260). coords(wk1gt_5,50,166,260).  visible(wk1gt_5,260). happensAt(inactive(wk1gt_5),260).
orientation(wk1gt_4,159,261). coords(wk1gt_4,48,179,261).  visible(wk1gt_4,261). happensAt(active(wk1gt_4),261). orientation(wk1gt_5,162,261). coords(wk1gt_5,50,166,261).  visible(wk1gt_5,261). happensAt(inactive(wk1gt_5),261).
orientation(wk1gt_4,160,262). coords(wk1gt_4,48,179,262).  visible(wk1gt_4,262). happensAt(active(wk1gt_4),262). orientation(wk1gt_5,162,262). coords(wk1gt_5,50,166,262).  visible(wk1gt_5,262). happensAt(inactive(wk1gt_5),262).
orientation(wk1gt_4,158,263). coords(wk1gt_4,48,179,263).  visible(wk1gt_4,263). happensAt(active(wk1gt_4),263). orientation(wk1gt_5,162,263). coords(wk1gt_5,50,166,263).  visible(wk1gt_5,263). happensAt(inactive(wk1gt_5),263).
orientation(wk1gt_4,162,264). coords(wk1gt_4,48,179,264).  visible(wk1gt_4,264). happensAt(active(wk1gt_4),264). orientation(wk1gt_5,162,264). coords(wk1gt_5,50,166,264).  visible(wk1gt_5,264). happensAt(inactive(wk1gt_5),264).
orientation(wk1gt_4,163,265). coords(wk1gt_4,48,179,265).  visible(wk1gt_4,265). happensAt(active(wk1gt_4),265). orientation(wk1gt_5,162,265). coords(wk1gt_5,50,166,265).  visible(wk1gt_5,265). happensAt(inactive(wk1gt_5),265).
orientation(wk1gt_4,154,266). coords(wk1gt_4,48,179,266).  visible(wk1gt_4,266). happensAt(active(wk1gt_4),266). orientation(wk1gt_5,162,266). coords(wk1gt_5,50,166,266).  visible(wk1gt_5,266). happensAt(inactive(wk1gt_5),266).
orientation(wk1gt_4,145,267). coords(wk1gt_4,48,179,267).  visible(wk1gt_4,267). happensAt(active(wk1gt_4),267). orientation(wk1gt_5,162,267). coords(wk1gt_5,50,166,267).  visible(wk1gt_5,267). happensAt(inactive(wk1gt_5),267).
orientation(wk1gt_4,152,268). coords(wk1gt_4,48,179,268).  visible(wk1gt_4,268). happensAt(active(wk1gt_4),268). orientation(wk1gt_5,162,268). coords(wk1gt_5,50,166,268).  visible(wk1gt_5,268). happensAt(inactive(wk1gt_5),268).
orientation(wk1gt_4,156,269). coords(wk1gt_4,48,178,269).  visible(wk1gt_4,269). happensAt(active(wk1gt_4),269). orientation(wk1gt_5,162,269). coords(wk1gt_5,50,166,269).  visible(wk1gt_5,269). happensAt(inactive(wk1gt_5),269).
orientation(wk1gt_4,158,270). coords(wk1gt_4,48,178,270).  visible(wk1gt_4,270). happensAt(active(wk1gt_4),270). orientation(wk1gt_5,162,270). coords(wk1gt_5,50,166,270).  visible(wk1gt_5,270). happensAt(inactive(wk1gt_5),270).
orientation(wk1gt_4,157,271). coords(wk1gt_4,49,178,271).  visible(wk1gt_4,271). happensAt(active(wk1gt_4),271). orientation(wk1gt_5,162,271). coords(wk1gt_5,50,166,271).  visible(wk1gt_5,271). happensAt(inactive(wk1gt_5),271).
orientation(wk1gt_4,153,272). coords(wk1gt_4,49,178,272).  visible(wk1gt_4,272). happensAt(active(wk1gt_4),272). orientation(wk1gt_5,162,272). coords(wk1gt_5,50,166,272).  visible(wk1gt_5,272). happensAt(active(wk1gt_5),272).
orientation(wk1gt_4,157,273). coords(wk1gt_4,49,178,273).  visible(wk1gt_4,273). happensAt(active(wk1gt_4),273). orientation(wk1gt_5,162,273). coords(wk1gt_5,50,166,273).  visible(wk1gt_5,273). happensAt(active(wk1gt_5),273).
orientation(wk1gt_4,156,274). coords(wk1gt_4,49,178,274).  visible(wk1gt_4,274). happensAt(active(wk1gt_4),274). orientation(wk1gt_5,162,274). coords(wk1gt_5,50,166,274).  visible(wk1gt_5,274). happensAt(active(wk1gt_5),274).
orientation(wk1gt_4,153,275). coords(wk1gt_4,49,178,275).  visible(wk1gt_4,275). happensAt(active(wk1gt_4),275). orientation(wk1gt_5,162,275). coords(wk1gt_5,50,166,275).  visible(wk1gt_5,275). happensAt(active(wk1gt_5),275).
orientation(wk1gt_4,154,276). coords(wk1gt_4,49,177,276).  visible(wk1gt_4,276). happensAt(active(wk1gt_4),276). orientation(wk1gt_5,162,276). coords(wk1gt_5,50,166,276).  visible(wk1gt_5,276). happensAt(active(wk1gt_5),276).
orientation(wk1gt_4,154,277). coords(wk1gt_4,49,177,277).  visible(wk1gt_4,277). happensAt(active(wk1gt_4),277). orientation(wk1gt_5,162,277). coords(wk1gt_5,50,166,277).  visible(wk1gt_5,277). happensAt(active(wk1gt_5),277).
orientation(wk1gt_4,156,278). coords(wk1gt_4,50,177,278).  visible(wk1gt_4,278). happensAt(active(wk1gt_4),278). orientation(wk1gt_5,162,278). coords(wk1gt_5,50,166,278).  visible(wk1gt_5,278). happensAt(active(wk1gt_5),278).
orientation(wk1gt_4,155,279). coords(wk1gt_4,51,177,279).  visible(wk1gt_4,279). happensAt(active(wk1gt_4),279). orientation(wk1gt_5,162,279). coords(wk1gt_5,50,166,279).  visible(wk1gt_5,279). happensAt(active(wk1gt_5),279).
#show holdsAt/2.