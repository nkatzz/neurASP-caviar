
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

	orientation(fomdgt3_0,179,24). coords(fomdgt3_0,244,232,24).  visible(fomdgt3_0,24). happensAt(walking(fomdgt3_0),24). orientation(fomdgt3_1,160,24). coords(fomdgt3_1,155,162,24).  visible(fomdgt3_1,24). happensAt(active(fomdgt3_1),24).
orientation(fomdgt3_0,179,25). coords(fomdgt3_0,245,235,25).  visible(fomdgt3_0,25). happensAt(walking(fomdgt3_0),25). orientation(fomdgt3_1,160,25). coords(fomdgt3_1,155,162,25).  visible(fomdgt3_1,25). happensAt(active(fomdgt3_1),25).
orientation(fomdgt3_0,179,26). coords(fomdgt3_0,250,239,26).  visible(fomdgt3_0,26). happensAt(walking(fomdgt3_0),26). orientation(fomdgt3_1,160,26). coords(fomdgt3_1,155,162,26).  visible(fomdgt3_1,26). happensAt(active(fomdgt3_1),26).
orientation(fomdgt3_0,179,27). coords(fomdgt3_0,253,241,27).  visible(fomdgt3_0,27). happensAt(walking(fomdgt3_0),27). orientation(fomdgt3_1,160,27). coords(fomdgt3_1,155,162,27).  visible(fomdgt3_1,27). happensAt(active(fomdgt3_1),27).
orientation(fomdgt3_0,10,28). coords(fomdgt3_0,256,243,28).  visible(fomdgt3_0,28). happensAt(walking(fomdgt3_0),28). orientation(fomdgt3_1,160,28). coords(fomdgt3_1,155,162,28).  visible(fomdgt3_1,28). happensAt(active(fomdgt3_1),28).
orientation(fomdgt3_0,20,29). coords(fomdgt3_0,258,243,29).  visible(fomdgt3_0,29). happensAt(walking(fomdgt3_0),29). orientation(fomdgt3_1,160,29). coords(fomdgt3_1,155,162,29).  visible(fomdgt3_1,29). happensAt(active(fomdgt3_1),29).
orientation(fomdgt3_0,20,30). coords(fomdgt3_0,260,245,30).  visible(fomdgt3_0,30). happensAt(walking(fomdgt3_0),30). orientation(fomdgt3_1,160,30). coords(fomdgt3_1,154,162,30).  visible(fomdgt3_1,30). happensAt(active(fomdgt3_1),30).
orientation(fomdgt3_0,20,31). coords(fomdgt3_0,260,247,31).  visible(fomdgt3_0,31). happensAt(walking(fomdgt3_0),31). orientation(fomdgt3_1,160,31). coords(fomdgt3_1,154,162,31).  visible(fomdgt3_1,31). happensAt(active(fomdgt3_1),31).
orientation(fomdgt3_0,30,32). coords(fomdgt3_0,263,250,32).  visible(fomdgt3_0,32). happensAt(walking(fomdgt3_0),32). orientation(fomdgt3_1,160,32). coords(fomdgt3_1,154,162,32).  visible(fomdgt3_1,32). happensAt(active(fomdgt3_1),32).
orientation(fomdgt3_0,40,33). coords(fomdgt3_0,264,254,33).  visible(fomdgt3_0,33). happensAt(walking(fomdgt3_0),33). orientation(fomdgt3_1,160,33). coords(fomdgt3_1,154,162,33).  visible(fomdgt3_1,33). happensAt(active(fomdgt3_1),33).
orientation(fomdgt3_0,40,34). coords(fomdgt3_0,266,258,34).  visible(fomdgt3_0,34). happensAt(walking(fomdgt3_0),34). orientation(fomdgt3_1,160,34). coords(fomdgt3_1,154,162,34).  visible(fomdgt3_1,34). happensAt(active(fomdgt3_1),34).
orientation(fomdgt3_0,40,35). coords(fomdgt3_0,266,262,35).  visible(fomdgt3_0,35). happensAt(walking(fomdgt3_0),35). orientation(fomdgt3_1,160,35). coords(fomdgt3_1,154,162,35).  visible(fomdgt3_1,35). happensAt(active(fomdgt3_1),35).
orientation(fomdgt3_0,40,36). coords(fomdgt3_0,269,266,36).  visible(fomdgt3_0,36). happensAt(walking(fomdgt3_0),36). orientation(fomdgt3_1,160,36). coords(fomdgt3_1,154,162,36).  visible(fomdgt3_1,36). happensAt(active(fomdgt3_1),36).
orientation(fomdgt3_0,40,37). coords(fomdgt3_0,272,269,37).  visible(fomdgt3_0,37). happensAt(walking(fomdgt3_0),37). orientation(fomdgt3_1,160,37). coords(fomdgt3_1,154,162,37).  visible(fomdgt3_1,37). happensAt(active(fomdgt3_1),37).
orientation(fomdgt3_0,40,38). coords(fomdgt3_0,275,269,38).  visible(fomdgt3_0,38). happensAt(walking(fomdgt3_0),38). orientation(fomdgt3_1,160,38). coords(fomdgt3_1,154,162,38).  visible(fomdgt3_1,38). happensAt(active(fomdgt3_1),38).
orientation(fomdgt3_0,40,39). coords(fomdgt3_0,278,270,39).  visible(fomdgt3_0,39). happensAt(walking(fomdgt3_0),39). orientation(fomdgt3_1,160,39). coords(fomdgt3_1,154,162,39).  visible(fomdgt3_1,39). happensAt(active(fomdgt3_1),39).
orientation(fomdgt3_0,40,40). coords(fomdgt3_0,280,270,40).  visible(fomdgt3_0,40). happensAt(walking(fomdgt3_0),40). orientation(fomdgt3_1,160,40). coords(fomdgt3_1,154,162,40).  visible(fomdgt3_1,40). happensAt(active(fomdgt3_1),40).
orientation(fomdgt3_0,40,41). coords(fomdgt3_0,282,270,41).  visible(fomdgt3_0,41). happensAt(walking(fomdgt3_0),41). orientation(fomdgt3_1,160,41). coords(fomdgt3_1,154,162,41).  visible(fomdgt3_1,41). happensAt(active(fomdgt3_1),41).
orientation(fomdgt3_0,50,42). coords(fomdgt3_0,282,270,42).  visible(fomdgt3_0,42). happensAt(walking(fomdgt3_0),42). orientation(fomdgt3_1,160,42). coords(fomdgt3_1,154,162,42).  visible(fomdgt3_1,42). happensAt(active(fomdgt3_1),42).
orientation(fomdgt3_0,50,43). coords(fomdgt3_0,286,271,43).  visible(fomdgt3_0,43). happensAt(walking(fomdgt3_0),43). orientation(fomdgt3_1,160,43). coords(fomdgt3_1,154,162,43).  visible(fomdgt3_1,43). happensAt(active(fomdgt3_1),43).
orientation(fomdgt3_0,50,44). coords(fomdgt3_0,289,273,44).  visible(fomdgt3_0,44). happensAt(walking(fomdgt3_0),44). orientation(fomdgt3_1,160,44). coords(fomdgt3_1,154,162,44).  visible(fomdgt3_1,44). happensAt(active(fomdgt3_1),44).
orientation(fomdgt3_0,50,45). coords(fomdgt3_0,291,275,45).  visible(fomdgt3_0,45). happensAt(walking(fomdgt3_0),45). orientation(fomdgt3_1,160,45). coords(fomdgt3_1,154,162,45).  visible(fomdgt3_1,45). happensAt(active(fomdgt3_1),45).
orientation(fomdgt3_0,75,46). coords(fomdgt3_0,293,277,46).  visible(fomdgt3_0,46). happensAt(walking(fomdgt3_0),46). orientation(fomdgt3_1,160,46). coords(fomdgt3_1,154,162,46).  visible(fomdgt3_1,46). happensAt(active(fomdgt3_1),46).
orientation(fomdgt3_0,75,47). coords(fomdgt3_0,295,280,47).  visible(fomdgt3_0,47). happensAt(walking(fomdgt3_0),47). orientation(fomdgt3_1,160,47). coords(fomdgt3_1,154,162,47).  visible(fomdgt3_1,47). happensAt(active(fomdgt3_1),47).
orientation(fomdgt3_0,75,48). coords(fomdgt3_0,297,283,48).  visible(fomdgt3_0,48). happensAt(walking(fomdgt3_0),48). orientation(fomdgt3_1,160,48). coords(fomdgt3_1,154,162,48).  visible(fomdgt3_1,48). happensAt(active(fomdgt3_1),48).
#show holdsAt/2.