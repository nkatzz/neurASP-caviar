
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

	orientation(spgt_0,156,72). coords(spgt_0,57,159,72).  visible(spgt_0,72). happensAt(walking(spgt_0),72). orientation(spgt_1,160,72). coords(spgt_1,57,149,72).  visible(spgt_1,72). happensAt(walking(spgt_1),72).
orientation(spgt_0,156,73). coords(spgt_0,58,159,73).  visible(spgt_0,73). happensAt(walking(spgt_0),73). orientation(spgt_1,160,73). coords(spgt_1,58,149,73).  visible(spgt_1,73). happensAt(walking(spgt_1),73).
orientation(spgt_0,156,74). coords(spgt_0,59,159,74).  visible(spgt_0,74). happensAt(walking(spgt_0),74). orientation(spgt_1,160,74). coords(spgt_1,59,149,74).  visible(spgt_1,74). happensAt(walking(spgt_1),74).
orientation(spgt_0,156,75). coords(spgt_0,60,159,75).  visible(spgt_0,75). happensAt(walking(spgt_0),75). orientation(spgt_1,160,75). coords(spgt_1,60,148,75).  visible(spgt_1,75). happensAt(walking(spgt_1),75).
orientation(spgt_0,156,76). coords(spgt_0,61,158,76).  visible(spgt_0,76). happensAt(walking(spgt_0),76). orientation(spgt_1,160,76). coords(spgt_1,60,148,76).  visible(spgt_1,76). happensAt(walking(spgt_1),76).
orientation(spgt_0,156,77). coords(spgt_0,61,158,77).  visible(spgt_0,77). happensAt(walking(spgt_0),77). orientation(spgt_1,160,77). coords(spgt_1,60,148,77).  visible(spgt_1,77). happensAt(walking(spgt_1),77).
orientation(spgt_0,156,78). coords(spgt_0,62,158,78).  visible(spgt_0,78). happensAt(walking(spgt_0),78). orientation(spgt_1,160,78). coords(spgt_1,61,147,78).  visible(spgt_1,78). happensAt(walking(spgt_1),78).
orientation(spgt_0,156,79). coords(spgt_0,62,157,79).  visible(spgt_0,79). happensAt(walking(spgt_0),79). orientation(spgt_1,160,79). coords(spgt_1,63,147,79).  visible(spgt_1,79). happensAt(walking(spgt_1),79).
orientation(spgt_0,156,80). coords(spgt_0,63,157,80).  visible(spgt_0,80). happensAt(walking(spgt_0),80). orientation(spgt_1,160,80). coords(spgt_1,63,147,80).  visible(spgt_1,80). happensAt(walking(spgt_1),80).
orientation(spgt_0,156,81). coords(spgt_0,63,157,81).  visible(spgt_0,81). happensAt(walking(spgt_0),81). orientation(spgt_1,160,81). coords(spgt_1,63,146,81).  visible(spgt_1,81). happensAt(walking(spgt_1),81).
orientation(spgt_0,156,82). coords(spgt_0,63,157,82).  visible(spgt_0,82). happensAt(walking(spgt_0),82). orientation(spgt_1,160,82). coords(spgt_1,63,146,82).  visible(spgt_1,82). happensAt(walking(spgt_1),82).
orientation(spgt_0,152,83). coords(spgt_0,64,156,83).  visible(spgt_0,83). happensAt(walking(spgt_0),83). orientation(spgt_1,160,83). coords(spgt_1,64,145,83).  visible(spgt_1,83). happensAt(walking(spgt_1),83).
orientation(spgt_0,152,84). coords(spgt_0,64,155,84).  visible(spgt_0,84). happensAt(walking(spgt_0),84). orientation(spgt_1,160,84). coords(spgt_1,64,145,84).  visible(spgt_1,84). happensAt(walking(spgt_1),84).
orientation(spgt_0,152,85). coords(spgt_0,64,155,85).  visible(spgt_0,85). happensAt(walking(spgt_0),85). orientation(spgt_1,160,85). coords(spgt_1,65,144,85).  visible(spgt_1,85). happensAt(walking(spgt_1),85).
orientation(spgt_0,152,86). coords(spgt_0,64,154,86).  visible(spgt_0,86). happensAt(walking(spgt_0),86). orientation(spgt_1,160,86). coords(spgt_1,65,144,86).  visible(spgt_1,86). happensAt(walking(spgt_1),86).
orientation(spgt_0,152,87). coords(spgt_0,65,154,87).  visible(spgt_0,87). happensAt(walking(spgt_0),87). orientation(spgt_1,160,87). coords(spgt_1,65,143,87).  visible(spgt_1,87). happensAt(walking(spgt_1),87).
orientation(spgt_0,152,88). coords(spgt_0,66,152,88).  visible(spgt_0,88). happensAt(walking(spgt_0),88). orientation(spgt_1,160,88). coords(spgt_1,66,143,88).  visible(spgt_1,88). happensAt(walking(spgt_1),88).
orientation(spgt_0,152,89). coords(spgt_0,67,152,89).  visible(spgt_0,89). happensAt(walking(spgt_0),89). orientation(spgt_1,160,89). coords(spgt_1,67,142,89).  visible(spgt_1,89). happensAt(walking(spgt_1),89).
orientation(spgt_0,152,90). coords(spgt_0,68,152,90).  visible(spgt_0,90). happensAt(walking(spgt_0),90). orientation(spgt_1,160,90). coords(spgt_1,68,142,90).  visible(spgt_1,90). happensAt(walking(spgt_1),90).
orientation(spgt_0,152,91). coords(spgt_0,68,152,91).  visible(spgt_0,91). happensAt(walking(spgt_0),91). orientation(spgt_1,160,91). coords(spgt_1,69,142,91).  visible(spgt_1,91). happensAt(walking(spgt_1),91).
orientation(spgt_0,152,92). coords(spgt_0,68,152,92).  visible(spgt_0,92). happensAt(walking(spgt_0),92). orientation(spgt_1,160,92). coords(spgt_1,69,142,92).  visible(spgt_1,92). happensAt(walking(spgt_1),92).
orientation(spgt_0,152,93). coords(spgt_0,70,151,93).  visible(spgt_0,93). happensAt(walking(spgt_0),93). orientation(spgt_1,154,93). coords(spgt_1,70,141,93).  visible(spgt_1,93). happensAt(walking(spgt_1),93).
orientation(spgt_0,152,94). coords(spgt_0,70,150,94).  visible(spgt_0,94). happensAt(walking(spgt_0),94). orientation(spgt_1,154,94). coords(spgt_1,71,140,94).  visible(spgt_1,94). happensAt(walking(spgt_1),94).
orientation(spgt_0,152,95). coords(spgt_0,70,150,95).  visible(spgt_0,95). happensAt(walking(spgt_0),95). orientation(spgt_1,154,95). coords(spgt_1,71,140,95).  visible(spgt_1,95). happensAt(walking(spgt_1),95).
orientation(spgt_0,152,96). coords(spgt_0,70,150,96).  visible(spgt_0,96). happensAt(walking(spgt_0),96). orientation(spgt_1,154,96). coords(spgt_1,72,140,96).  visible(spgt_1,96). happensAt(walking(spgt_1),96).
#show holdsAt/2.