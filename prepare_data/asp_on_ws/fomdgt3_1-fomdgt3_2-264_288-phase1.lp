
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

	orientation(fomdgt3_1,150,264). coords(fomdgt3_1,59,96,264).  visible(fomdgt3_1,264). happensAt(walking(fomdgt3_1),264). orientation(fomdgt3_2,130,264). coords(fomdgt3_2,237,140,264).  visible(fomdgt3_2,264). happensAt(walking(fomdgt3_2),264).
orientation(fomdgt3_1,150,265). coords(fomdgt3_1,59,96,265).  visible(fomdgt3_1,265). happensAt(walking(fomdgt3_1),265). orientation(fomdgt3_2,130,265). coords(fomdgt3_2,237,140,265).  visible(fomdgt3_2,265). happensAt(walking(fomdgt3_2),265).
orientation(fomdgt3_1,150,266). coords(fomdgt3_1,59,96,266).  visible(fomdgt3_1,266). happensAt(walking(fomdgt3_1),266). orientation(fomdgt3_2,130,266). coords(fomdgt3_2,237,140,266).  visible(fomdgt3_2,266). happensAt(walking(fomdgt3_2),266).
orientation(fomdgt3_1,150,267). coords(fomdgt3_1,58,96,267).  visible(fomdgt3_1,267). happensAt(walking(fomdgt3_1),267). orientation(fomdgt3_2,130,267). coords(fomdgt3_2,234,140,267).  visible(fomdgt3_2,267). happensAt(walking(fomdgt3_2),267).
orientation(fomdgt3_1,150,268). coords(fomdgt3_1,58,96,268).  visible(fomdgt3_1,268). happensAt(walking(fomdgt3_1),268). orientation(fomdgt3_2,130,268). coords(fomdgt3_2,234,140,268).  visible(fomdgt3_2,268). happensAt(walking(fomdgt3_2),268).
orientation(fomdgt3_1,150,269). coords(fomdgt3_1,57,96,269).  visible(fomdgt3_1,269). happensAt(walking(fomdgt3_1),269). orientation(fomdgt3_2,130,269). coords(fomdgt3_2,235,139,269).  visible(fomdgt3_2,269). happensAt(walking(fomdgt3_2),269).
orientation(fomdgt3_1,150,270). coords(fomdgt3_1,57,96,270).  visible(fomdgt3_1,270). happensAt(walking(fomdgt3_1),270). orientation(fomdgt3_2,130,270). coords(fomdgt3_2,235,135,270).  visible(fomdgt3_2,270). happensAt(walking(fomdgt3_2),270).
orientation(fomdgt3_1,150,271). coords(fomdgt3_1,57,96,271).  visible(fomdgt3_1,271). happensAt(walking(fomdgt3_1),271). orientation(fomdgt3_2,130,271). coords(fomdgt3_2,235,135,271).  visible(fomdgt3_2,271). happensAt(walking(fomdgt3_2),271).
orientation(fomdgt3_1,150,272). coords(fomdgt3_1,57,96,272).  visible(fomdgt3_1,272). happensAt(walking(fomdgt3_1),272). orientation(fomdgt3_2,130,272). coords(fomdgt3_2,235,135,272).  visible(fomdgt3_2,272). happensAt(walking(fomdgt3_2),272).
orientation(fomdgt3_1,150,273). coords(fomdgt3_1,53,96,273).  visible(fomdgt3_1,273). happensAt(walking(fomdgt3_1),273). orientation(fomdgt3_2,130,273). coords(fomdgt3_2,235,133,273).  visible(fomdgt3_2,273). happensAt(walking(fomdgt3_2),273).
orientation(fomdgt3_1,150,274). coords(fomdgt3_1,53,96,274).  visible(fomdgt3_1,274). happensAt(walking(fomdgt3_1),274). orientation(fomdgt3_2,130,274). coords(fomdgt3_2,236,133,274).  visible(fomdgt3_2,274). happensAt(walking(fomdgt3_2),274).
orientation(fomdgt3_1,150,275). coords(fomdgt3_1,53,96,275).  visible(fomdgt3_1,275). happensAt(walking(fomdgt3_1),275). orientation(fomdgt3_2,130,275). coords(fomdgt3_2,236,133,275).  visible(fomdgt3_2,275). happensAt(walking(fomdgt3_2),275).
orientation(fomdgt3_1,150,276). coords(fomdgt3_1,53,96,276).  visible(fomdgt3_1,276). happensAt(walking(fomdgt3_1),276). orientation(fomdgt3_2,130,276). coords(fomdgt3_2,235,132,276).  visible(fomdgt3_2,276). happensAt(walking(fomdgt3_2),276).
orientation(fomdgt3_1,150,277). coords(fomdgt3_1,51,96,277).  visible(fomdgt3_1,277). happensAt(walking(fomdgt3_1),277). orientation(fomdgt3_2,130,277). coords(fomdgt3_2,235,130,277).  visible(fomdgt3_2,277). happensAt(walking(fomdgt3_2),277).
orientation(fomdgt3_1,150,278). coords(fomdgt3_1,51,96,278).  visible(fomdgt3_1,278). happensAt(walking(fomdgt3_1),278). orientation(fomdgt3_2,130,278). coords(fomdgt3_2,234,129,278).  visible(fomdgt3_2,278). happensAt(walking(fomdgt3_2),278).
orientation(fomdgt3_1,150,279). coords(fomdgt3_1,51,96,279).  visible(fomdgt3_1,279). happensAt(walking(fomdgt3_1),279). orientation(fomdgt3_2,130,279). coords(fomdgt3_2,234,129,279).  visible(fomdgt3_2,279). happensAt(walking(fomdgt3_2),279).
orientation(fomdgt3_1,150,280). coords(fomdgt3_1,51,96,280).  visible(fomdgt3_1,280). happensAt(walking(fomdgt3_1),280). orientation(fomdgt3_2,130,280). coords(fomdgt3_2,234,128,280).  visible(fomdgt3_2,280). happensAt(walking(fomdgt3_2),280).
orientation(fomdgt3_1,150,281). coords(fomdgt3_1,51,96,281).  visible(fomdgt3_1,281). happensAt(walking(fomdgt3_1),281). orientation(fomdgt3_2,130,281). coords(fomdgt3_2,232,127,281).  visible(fomdgt3_2,281). happensAt(walking(fomdgt3_2),281).
orientation(fomdgt3_1,150,282). coords(fomdgt3_1,51,96,282).  visible(fomdgt3_1,282). happensAt(walking(fomdgt3_1),282). orientation(fomdgt3_2,130,282). coords(fomdgt3_2,227,127,282).  visible(fomdgt3_2,282). happensAt(walking(fomdgt3_2),282).
orientation(fomdgt3_1,150,283). coords(fomdgt3_1,51,96,283).  visible(fomdgt3_1,283). happensAt(walking(fomdgt3_1),283). orientation(fomdgt3_2,130,283). coords(fomdgt3_2,227,127,283).  visible(fomdgt3_2,283). happensAt(walking(fomdgt3_2),283).
orientation(fomdgt3_1,150,284). coords(fomdgt3_1,51,96,284).  visible(fomdgt3_1,284). happensAt(walking(fomdgt3_1),284). orientation(fomdgt3_2,130,284). coords(fomdgt3_2,224,127,284).  visible(fomdgt3_2,284). happensAt(walking(fomdgt3_2),284).
orientation(fomdgt3_1,150,285). coords(fomdgt3_1,51,96,285).  visible(fomdgt3_1,285). happensAt(walking(fomdgt3_1),285). orientation(fomdgt3_2,130,285). coords(fomdgt3_2,224,127,285).  visible(fomdgt3_2,285). happensAt(walking(fomdgt3_2),285).
orientation(fomdgt3_1,150,286). coords(fomdgt3_1,51,96,286).  visible(fomdgt3_1,286). happensAt(active(fomdgt3_1),286). orientation(fomdgt3_2,130,286). coords(fomdgt3_2,224,125,286).  visible(fomdgt3_2,286). happensAt(walking(fomdgt3_2),286).
orientation(fomdgt3_1,150,287). coords(fomdgt3_1,51,96,287).  visible(fomdgt3_1,287). happensAt(active(fomdgt3_1),287). orientation(fomdgt3_2,130,287). coords(fomdgt3_2,224,123,287).  visible(fomdgt3_2,287). happensAt(walking(fomdgt3_2),287).
orientation(fomdgt3_1,150,288). coords(fomdgt3_1,51,96,288).  visible(fomdgt3_1,288). happensAt(active(fomdgt3_1),288). orientation(fomdgt3_2,130,288). coords(fomdgt3_2,224,122,288).  visible(fomdgt3_2,288). happensAt(walking(fomdgt3_2),288).
#show holdsAt/2.