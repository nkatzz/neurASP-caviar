
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

	orientation(fra2gt_0,150,72). coords(fra2gt_0,157,129,72).  visible(fra2gt_0,72). happensAt(active(fra2gt_0),72). orientation(fra2gt_5,150,72). coords(fra2gt_5,32,112,72).  visible(fra2gt_5,72). happensAt(walking(fra2gt_5),72).
orientation(fra2gt_0,150,73). coords(fra2gt_0,157,129,73).  visible(fra2gt_0,73). happensAt(active(fra2gt_0),73). orientation(fra2gt_5,150,73). coords(fra2gt_5,32,113,73).  visible(fra2gt_5,73). happensAt(walking(fra2gt_5),73).
orientation(fra2gt_0,150,74). coords(fra2gt_0,157,129,74).  visible(fra2gt_0,74). happensAt(active(fra2gt_0),74). orientation(fra2gt_5,150,74). coords(fra2gt_5,32,113,74).  visible(fra2gt_5,74). happensAt(walking(fra2gt_5),74).
orientation(fra2gt_0,150,75). coords(fra2gt_0,158,129,75).  visible(fra2gt_0,75). happensAt(active(fra2gt_0),75). orientation(fra2gt_5,150,75). coords(fra2gt_5,32,113,75).  visible(fra2gt_5,75). happensAt(walking(fra2gt_5),75).
orientation(fra2gt_0,150,76). coords(fra2gt_0,158,129,76).  visible(fra2gt_0,76). happensAt(active(fra2gt_0),76). orientation(fra2gt_5,150,76). coords(fra2gt_5,32,113,76).  visible(fra2gt_5,76). happensAt(walking(fra2gt_5),76).
orientation(fra2gt_0,150,77). coords(fra2gt_0,160,130,77).  visible(fra2gt_0,77). happensAt(active(fra2gt_0),77). orientation(fra2gt_5,150,77). coords(fra2gt_5,32,115,77).  visible(fra2gt_5,77). happensAt(walking(fra2gt_5),77).
orientation(fra2gt_0,150,78). coords(fra2gt_0,161,130,78).  visible(fra2gt_0,78). happensAt(active(fra2gt_0),78). orientation(fra2gt_5,150,78). coords(fra2gt_5,32,115,78).  visible(fra2gt_5,78). happensAt(walking(fra2gt_5),78).
orientation(fra2gt_0,150,79). coords(fra2gt_0,163,130,79).  visible(fra2gt_0,79). happensAt(active(fra2gt_0),79). orientation(fra2gt_5,150,79). coords(fra2gt_5,32,115,79).  visible(fra2gt_5,79). happensAt(walking(fra2gt_5),79).
orientation(fra2gt_0,150,80). coords(fra2gt_0,165,131,80).  visible(fra2gt_0,80). happensAt(active(fra2gt_0),80). orientation(fra2gt_5,150,80). coords(fra2gt_5,32,116,80).  visible(fra2gt_5,80). happensAt(walking(fra2gt_5),80).
orientation(fra2gt_0,150,81). coords(fra2gt_0,166,131,81).  visible(fra2gt_0,81). happensAt(active(fra2gt_0),81). orientation(fra2gt_5,150,81). coords(fra2gt_5,32,116,81).  visible(fra2gt_5,81). happensAt(walking(fra2gt_5),81).
orientation(fra2gt_0,150,82). coords(fra2gt_0,168,131,82).  visible(fra2gt_0,82). happensAt(walking(fra2gt_0),82). orientation(fra2gt_5,150,82). coords(fra2gt_5,32,116,82).  visible(fra2gt_5,82). happensAt(walking(fra2gt_5),82).
orientation(fra2gt_0,150,83). coords(fra2gt_0,170,132,83).  visible(fra2gt_0,83). happensAt(walking(fra2gt_0),83). orientation(fra2gt_5,150,83). coords(fra2gt_5,32,116,83).  visible(fra2gt_5,83). happensAt(walking(fra2gt_5),83).
orientation(fra2gt_0,150,84). coords(fra2gt_0,173,132,84).  visible(fra2gt_0,84). happensAt(walking(fra2gt_0),84). orientation(fra2gt_5,150,84). coords(fra2gt_5,32,116,84).  visible(fra2gt_5,84). happensAt(walking(fra2gt_5),84).
orientation(fra2gt_0,150,85). coords(fra2gt_0,175,132,85).  visible(fra2gt_0,85). happensAt(walking(fra2gt_0),85). orientation(fra2gt_5,150,85). coords(fra2gt_5,31,116,85).  visible(fra2gt_5,85). happensAt(walking(fra2gt_5),85).
orientation(fra2gt_0,150,86). coords(fra2gt_0,178,133,86).  visible(fra2gt_0,86). happensAt(walking(fra2gt_0),86). orientation(fra2gt_5,150,86). coords(fra2gt_5,31,116,86).  visible(fra2gt_5,86). happensAt(walking(fra2gt_5),86).
orientation(fra2gt_0,150,87). coords(fra2gt_0,180,133,87).  visible(fra2gt_0,87). happensAt(walking(fra2gt_0),87). orientation(fra2gt_5,150,87). coords(fra2gt_5,31,116,87).  visible(fra2gt_5,87). happensAt(walking(fra2gt_5),87).
orientation(fra2gt_0,150,88). coords(fra2gt_0,181,134,88).  visible(fra2gt_0,88). happensAt(walking(fra2gt_0),88). orientation(fra2gt_5,150,88). coords(fra2gt_5,31,117,88).  visible(fra2gt_5,88). happensAt(walking(fra2gt_5),88).
orientation(fra2gt_0,150,89). coords(fra2gt_0,182,135,89).  visible(fra2gt_0,89). happensAt(walking(fra2gt_0),89). orientation(fra2gt_5,150,89). coords(fra2gt_5,31,117,89).  visible(fra2gt_5,89). happensAt(walking(fra2gt_5),89).
orientation(fra2gt_0,150,90). coords(fra2gt_0,183,135,90).  visible(fra2gt_0,90). happensAt(walking(fra2gt_0),90). orientation(fra2gt_5,150,90). coords(fra2gt_5,30,117,90).  visible(fra2gt_5,90). happensAt(walking(fra2gt_5),90).
orientation(fra2gt_0,150,91). coords(fra2gt_0,183,137,91).  visible(fra2gt_0,91). happensAt(walking(fra2gt_0),91). orientation(fra2gt_5,150,91). coords(fra2gt_5,30,117,91).  visible(fra2gt_5,91). happensAt(walking(fra2gt_5),91).
orientation(fra2gt_0,150,92). coords(fra2gt_0,184,140,92).  visible(fra2gt_0,92). happensAt(walking(fra2gt_0),92). orientation(fra2gt_5,150,92). coords(fra2gt_5,30,117,92).  visible(fra2gt_5,92). happensAt(walking(fra2gt_5),92).
orientation(fra2gt_0,150,93). coords(fra2gt_0,188,145,93).  visible(fra2gt_0,93). happensAt(walking(fra2gt_0),93). orientation(fra2gt_5,150,93). coords(fra2gt_5,29,119,93).  visible(fra2gt_5,93). happensAt(walking(fra2gt_5),93).
orientation(fra2gt_0,150,94). coords(fra2gt_0,191,150,94).  visible(fra2gt_0,94). happensAt(walking(fra2gt_0),94). orientation(fra2gt_5,150,94). coords(fra2gt_5,29,120,94).  visible(fra2gt_5,94). happensAt(walking(fra2gt_5),94).
orientation(fra2gt_0,150,95). coords(fra2gt_0,194,155,95).  visible(fra2gt_0,95). happensAt(walking(fra2gt_0),95). orientation(fra2gt_5,150,95). coords(fra2gt_5,29,120,95).  visible(fra2gt_5,95). happensAt(walking(fra2gt_5),95).
orientation(fra2gt_0,150,96). coords(fra2gt_0,196,159,96).  visible(fra2gt_0,96). happensAt(walking(fra2gt_0),96). orientation(fra2gt_5,150,96). coords(fra2gt_5,30,121,96).  visible(fra2gt_5,96). happensAt(walking(fra2gt_5),96).
#show holdsAt/2.