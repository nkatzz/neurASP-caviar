
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

	orientation(mwt2gt_1,138,253). coords(mwt2gt_1,139,71,253).  visible(mwt2gt_1,253). happensAt(walking(mwt2gt_1),253). orientation(mwt2gt_2,145,253). coords(mwt2gt_2,196,145,253).  visible(mwt2gt_2,253). happensAt(walking(mwt2gt_2),253).
orientation(mwt2gt_1,138,254). coords(mwt2gt_1,139,72,254).  visible(mwt2gt_1,254). happensAt(walking(mwt2gt_1),254). orientation(mwt2gt_2,145,254). coords(mwt2gt_2,194,143,254).  visible(mwt2gt_2,254). happensAt(walking(mwt2gt_2),254).
orientation(mwt2gt_1,138,255). coords(mwt2gt_1,139,73,255).  visible(mwt2gt_1,255). happensAt(walking(mwt2gt_1),255). orientation(mwt2gt_2,141,255). coords(mwt2gt_2,192,140,255).  visible(mwt2gt_2,255). happensAt(walking(mwt2gt_2),255).
orientation(mwt2gt_1,138,256). coords(mwt2gt_1,139,73,256).  visible(mwt2gt_1,256). happensAt(walking(mwt2gt_1),256). orientation(mwt2gt_2,141,256). coords(mwt2gt_2,190,137,256).  visible(mwt2gt_2,256). happensAt(walking(mwt2gt_2),256).
orientation(mwt2gt_1,138,257). coords(mwt2gt_1,139,74,257).  visible(mwt2gt_1,257). happensAt(walking(mwt2gt_1),257). orientation(mwt2gt_2,141,257). coords(mwt2gt_2,190,135,257).  visible(mwt2gt_2,257). happensAt(walking(mwt2gt_2),257).
orientation(mwt2gt_1,138,258). coords(mwt2gt_1,139,74,258).  visible(mwt2gt_1,258). happensAt(walking(mwt2gt_1),258). orientation(mwt2gt_2,141,258). coords(mwt2gt_2,189,133,258).  visible(mwt2gt_2,258). happensAt(walking(mwt2gt_2),258).
orientation(mwt2gt_1,143,259). coords(mwt2gt_1,139,74,259).  visible(mwt2gt_1,259). happensAt(walking(mwt2gt_1),259). orientation(mwt2gt_2,141,259). coords(mwt2gt_2,189,132,259).  visible(mwt2gt_2,259). happensAt(walking(mwt2gt_2),259).
orientation(mwt2gt_1,143,260). coords(mwt2gt_1,139,74,260).  visible(mwt2gt_1,260). happensAt(walking(mwt2gt_1),260). orientation(mwt2gt_2,141,260). coords(mwt2gt_2,188,132,260).  visible(mwt2gt_2,260). happensAt(walking(mwt2gt_2),260).
orientation(mwt2gt_1,143,261). coords(mwt2gt_1,139,74,261).  visible(mwt2gt_1,261). happensAt(walking(mwt2gt_1),261). orientation(mwt2gt_2,145,261). coords(mwt2gt_2,188,131,261).  visible(mwt2gt_2,261). happensAt(walking(mwt2gt_2),261).
orientation(mwt2gt_1,143,262). coords(mwt2gt_1,139,74,262).  visible(mwt2gt_1,262). happensAt(walking(mwt2gt_1),262). orientation(mwt2gt_2,145,262). coords(mwt2gt_2,188,131,262).  visible(mwt2gt_2,262). happensAt(walking(mwt2gt_2),262).
orientation(mwt2gt_1,143,263). coords(mwt2gt_1,138,74,263).  visible(mwt2gt_1,263). happensAt(walking(mwt2gt_1),263). orientation(mwt2gt_2,145,263). coords(mwt2gt_2,187,130,263).  visible(mwt2gt_2,263). happensAt(walking(mwt2gt_2),263).
orientation(mwt2gt_1,143,264). coords(mwt2gt_1,138,75,264).  visible(mwt2gt_1,264). happensAt(walking(mwt2gt_1),264). orientation(mwt2gt_2,145,264). coords(mwt2gt_2,185,129,264).  visible(mwt2gt_2,264). happensAt(walking(mwt2gt_2),264).
orientation(mwt2gt_1,143,265). coords(mwt2gt_1,138,75,265).  visible(mwt2gt_1,265). happensAt(walking(mwt2gt_1),265). orientation(mwt2gt_2,145,265). coords(mwt2gt_2,184,128,265).  visible(mwt2gt_2,265). happensAt(walking(mwt2gt_2),265).
orientation(mwt2gt_1,143,266). coords(mwt2gt_1,138,75,266).  visible(mwt2gt_1,266). happensAt(walking(mwt2gt_1),266). orientation(mwt2gt_2,145,266). coords(mwt2gt_2,183,126,266).  visible(mwt2gt_2,266). happensAt(walking(mwt2gt_2),266).
orientation(mwt2gt_1,143,267). coords(mwt2gt_1,138,75,267).  visible(mwt2gt_1,267). happensAt(walking(mwt2gt_1),267). orientation(mwt2gt_2,145,267). coords(mwt2gt_2,182,125,267).  visible(mwt2gt_2,267). happensAt(walking(mwt2gt_2),267).
orientation(mwt2gt_1,143,268). coords(mwt2gt_1,138,75,268).  visible(mwt2gt_1,268). happensAt(walking(mwt2gt_1),268). orientation(mwt2gt_2,145,268). coords(mwt2gt_2,179,123,268).  visible(mwt2gt_2,268). happensAt(walking(mwt2gt_2),268).
orientation(mwt2gt_1,143,269). coords(mwt2gt_1,138,75,269).  visible(mwt2gt_1,269). happensAt(walking(mwt2gt_1),269). orientation(mwt2gt_2,145,269). coords(mwt2gt_2,177,121,269).  visible(mwt2gt_2,269). happensAt(walking(mwt2gt_2),269).
orientation(mwt2gt_1,143,270). coords(mwt2gt_1,138,75,270).  visible(mwt2gt_1,270). happensAt(walking(mwt2gt_1),270). orientation(mwt2gt_2,145,270). coords(mwt2gt_2,176,120,270).  visible(mwt2gt_2,270). happensAt(walking(mwt2gt_2),270).
orientation(mwt2gt_1,143,271). coords(mwt2gt_1,138,75,271).  visible(mwt2gt_1,271). happensAt(walking(mwt2gt_1),271). orientation(mwt2gt_2,145,271). coords(mwt2gt_2,176,120,271).  visible(mwt2gt_2,271). happensAt(walking(mwt2gt_2),271).
orientation(mwt2gt_1,143,272). coords(mwt2gt_1,138,75,272).  visible(mwt2gt_1,272). happensAt(walking(mwt2gt_1),272). orientation(mwt2gt_2,145,272). coords(mwt2gt_2,174,120,272).  visible(mwt2gt_2,272). happensAt(walking(mwt2gt_2),272).
orientation(mwt2gt_1,143,273). coords(mwt2gt_1,138,75,273).  visible(mwt2gt_1,273). happensAt(walking(mwt2gt_1),273). orientation(mwt2gt_2,145,273). coords(mwt2gt_2,174,120,273).  visible(mwt2gt_2,273). happensAt(walking(mwt2gt_2),273).
orientation(mwt2gt_1,143,274). coords(mwt2gt_1,138,75,274).  visible(mwt2gt_1,274). happensAt(walking(mwt2gt_1),274). orientation(mwt2gt_2,145,274). coords(mwt2gt_2,173,119,274).  visible(mwt2gt_2,274). happensAt(walking(mwt2gt_2),274).
orientation(mwt2gt_1,143,275). coords(mwt2gt_1,138,75,275).  visible(mwt2gt_1,275). happensAt(walking(mwt2gt_1),275). orientation(mwt2gt_2,145,275). coords(mwt2gt_2,173,119,275).  visible(mwt2gt_2,275). happensAt(walking(mwt2gt_2),275).
orientation(mwt2gt_1,143,276). coords(mwt2gt_1,138,75,276).  visible(mwt2gt_1,276). happensAt(walking(mwt2gt_1),276). orientation(mwt2gt_2,145,276). coords(mwt2gt_2,172,118,276).  visible(mwt2gt_2,276). happensAt(walking(mwt2gt_2),276).
orientation(mwt2gt_1,143,277). coords(mwt2gt_1,138,75,277).  visible(mwt2gt_1,277). happensAt(walking(mwt2gt_1),277). orientation(mwt2gt_2,145,277). coords(mwt2gt_2,171,117,277).  visible(mwt2gt_2,277). happensAt(walking(mwt2gt_2),277).
#show holdsAt/2.