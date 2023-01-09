
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

	orientation(spgt_0,148,252). coords(spgt_0,123,110,252).  visible(spgt_0,252). happensAt(walking(spgt_0),252). orientation(spgt_1,137,252). coords(spgt_1,119,86,252).  visible(spgt_1,252). happensAt(walking(spgt_1),252).
orientation(spgt_0,148,253). coords(spgt_0,124,110,253).  visible(spgt_0,253). happensAt(walking(spgt_0),253). orientation(spgt_1,137,253). coords(spgt_1,120,85,253).  visible(spgt_1,253). happensAt(walking(spgt_1),253).
orientation(spgt_0,148,254). coords(spgt_0,125,110,254).  visible(spgt_0,254). happensAt(walking(spgt_0),254). orientation(spgt_1,137,254). coords(spgt_1,120,85,254).  visible(spgt_1,254). happensAt(walking(spgt_1),254).
orientation(spgt_0,148,255). coords(spgt_0,127,110,255).  visible(spgt_0,255). happensAt(walking(spgt_0),255). orientation(spgt_1,137,255). coords(spgt_1,121,85,255).  visible(spgt_1,255). happensAt(walking(spgt_1),255).
orientation(spgt_0,148,256). coords(spgt_0,128,110,256).  visible(spgt_0,256). happensAt(walking(spgt_0),256). orientation(spgt_1,137,256). coords(spgt_1,121,84,256).  visible(spgt_1,256). happensAt(walking(spgt_1),256).
orientation(spgt_0,148,257). coords(spgt_0,129,109,257).  visible(spgt_0,257). happensAt(walking(spgt_0),257). orientation(spgt_1,137,257). coords(spgt_1,122,83,257).  visible(spgt_1,257). happensAt(walking(spgt_1),257).
orientation(spgt_0,148,258). coords(spgt_0,130,109,258).  visible(spgt_0,258). happensAt(walking(spgt_0),258). orientation(spgt_1,137,258). coords(spgt_1,122,82,258).  visible(spgt_1,258). happensAt(walking(spgt_1),258).
orientation(spgt_0,148,259). coords(spgt_0,130,108,259).  visible(spgt_0,259). happensAt(walking(spgt_0),259). orientation(spgt_1,137,259). coords(spgt_1,122,81,259).  visible(spgt_1,259). happensAt(walking(spgt_1),259).
orientation(spgt_0,148,260). coords(spgt_0,131,108,260).  visible(spgt_0,260). happensAt(walking(spgt_0),260). orientation(spgt_1,137,260). coords(spgt_1,122,80,260).  visible(spgt_1,260). happensAt(walking(spgt_1),260).
orientation(spgt_0,148,261). coords(spgt_0,131,107,261).  visible(spgt_0,261). happensAt(walking(spgt_0),261). orientation(spgt_1,137,261). coords(spgt_1,122,78,261).  visible(spgt_1,261). happensAt(walking(spgt_1),261).
orientation(spgt_0,148,262). coords(spgt_0,132,107,262).  visible(spgt_0,262). happensAt(walking(spgt_0),262). orientation(spgt_1,137,262). coords(spgt_1,123,77,262).  visible(spgt_1,262). happensAt(walking(spgt_1),262).
orientation(spgt_0,148,263). coords(spgt_0,132,106,263).  visible(spgt_0,263). happensAt(walking(spgt_0),263). orientation(spgt_1,137,263). coords(spgt_1,123,75,263).  visible(spgt_1,263). happensAt(walking(spgt_1),263).
orientation(spgt_0,148,264). coords(spgt_0,132,105,264).  visible(spgt_0,264). happensAt(walking(spgt_0),264). orientation(spgt_1,137,264). coords(spgt_1,123,73,264).  visible(spgt_1,264). happensAt(walking(spgt_1),264).
orientation(spgt_0,145,265). coords(spgt_0,133,105,265).  visible(spgt_0,265). happensAt(walking(spgt_0),265). orientation(spgt_1,137,265). coords(spgt_1,125,73,265).  visible(spgt_1,265). happensAt(walking(spgt_1),265).
orientation(spgt_0,145,266). coords(spgt_0,133,105,266).  visible(spgt_0,266). happensAt(walking(spgt_0),266). orientation(spgt_1,137,266). coords(spgt_1,126,72,266).  visible(spgt_1,266). happensAt(walking(spgt_1),266).
orientation(spgt_0,147,267). coords(spgt_0,135,105,267).  visible(spgt_0,267). happensAt(walking(spgt_0),267). orientation(spgt_1,137,267). coords(spgt_1,126,71,267).  visible(spgt_1,267). happensAt(walking(spgt_1),267).
orientation(spgt_0,147,268). coords(spgt_0,137,106,268).  visible(spgt_0,268). happensAt(walking(spgt_0),268). orientation(spgt_1,137,268). coords(spgt_1,126,71,268).  visible(spgt_1,268). happensAt(walking(spgt_1),268).
orientation(spgt_0,147,269). coords(spgt_0,138,106,269).  visible(spgt_0,269). happensAt(walking(spgt_0),269). orientation(spgt_1,137,269). coords(spgt_1,127,71,269).  visible(spgt_1,269). happensAt(walking(spgt_1),269).
orientation(spgt_0,149,270). coords(spgt_0,139,106,270).  visible(spgt_0,270). happensAt(walking(spgt_0),270). orientation(spgt_1,137,270). coords(spgt_1,127,70,270).  visible(spgt_1,270). happensAt(walking(spgt_1),270).
orientation(spgt_0,149,271). coords(spgt_0,140,106,271).  visible(spgt_0,271). happensAt(walking(spgt_0),271). orientation(spgt_1,137,271). coords(spgt_1,128,69,271).  visible(spgt_1,271). happensAt(walking(spgt_1),271).
orientation(spgt_0,149,272). coords(spgt_0,141,106,272).  visible(spgt_0,272). happensAt(walking(spgt_0),272). orientation(spgt_1,137,272). coords(spgt_1,128,68,272).  visible(spgt_1,272). happensAt(walking(spgt_1),272).
orientation(spgt_0,149,273). coords(spgt_0,142,106,273).  visible(spgt_0,273). happensAt(walking(spgt_0),273). orientation(spgt_1,137,273). coords(spgt_1,128,66,273).  visible(spgt_1,273). happensAt(walking(spgt_1),273).
orientation(spgt_0,150,274). coords(spgt_0,142,106,274).  visible(spgt_0,274). happensAt(walking(spgt_0),274). orientation(spgt_1,137,274). coords(spgt_1,128,65,274).  visible(spgt_1,274). happensAt(walking(spgt_1),274).
orientation(spgt_0,149,275). coords(spgt_0,143,106,275).  visible(spgt_0,275). happensAt(walking(spgt_0),275). orientation(spgt_1,137,275). coords(spgt_1,128,63,275).  visible(spgt_1,275). happensAt(walking(spgt_1),275).
orientation(spgt_0,147,276). coords(spgt_0,143,106,276).  visible(spgt_0,276). happensAt(walking(spgt_0),276). orientation(spgt_1,137,276). coords(spgt_1,129,62,276).  visible(spgt_1,276). happensAt(walking(spgt_1),276).
#show holdsAt/2.