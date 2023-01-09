
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

	orientation(fomdgt3_0,170,12). coords(fomdgt3_0,221,200,12).  visible(fomdgt3_0,12). happensAt(walking(fomdgt3_0),12). orientation(fomdgt3_1,160,12). coords(fomdgt3_1,154,162,12).  visible(fomdgt3_1,12). happensAt(active(fomdgt3_1),12).
orientation(fomdgt3_0,170,13). coords(fomdgt3_0,224,204,13).  visible(fomdgt3_0,13). happensAt(walking(fomdgt3_0),13). orientation(fomdgt3_1,160,13). coords(fomdgt3_1,154,162,13).  visible(fomdgt3_1,13). happensAt(active(fomdgt3_1),13).
orientation(fomdgt3_0,170,14). coords(fomdgt3_0,228,207,14).  visible(fomdgt3_0,14). happensAt(walking(fomdgt3_0),14). orientation(fomdgt3_1,160,14). coords(fomdgt3_1,155,162,14).  visible(fomdgt3_1,14). happensAt(active(fomdgt3_1),14).
orientation(fomdgt3_0,170,15). coords(fomdgt3_0,231,209,15).  visible(fomdgt3_0,15). happensAt(walking(fomdgt3_0),15). orientation(fomdgt3_1,160,15). coords(fomdgt3_1,155,162,15).  visible(fomdgt3_1,15). happensAt(active(fomdgt3_1),15).
orientation(fomdgt3_0,170,16). coords(fomdgt3_0,234,213,16).  visible(fomdgt3_0,16). happensAt(walking(fomdgt3_0),16). orientation(fomdgt3_1,160,16). coords(fomdgt3_1,155,162,16).  visible(fomdgt3_1,16). happensAt(active(fomdgt3_1),16).
orientation(fomdgt3_0,179,17). coords(fomdgt3_0,236,214,17).  visible(fomdgt3_0,17). happensAt(walking(fomdgt3_0),17). orientation(fomdgt3_1,160,17). coords(fomdgt3_1,155,162,17).  visible(fomdgt3_1,17). happensAt(active(fomdgt3_1),17).
orientation(fomdgt3_0,179,18). coords(fomdgt3_0,238,216,18).  visible(fomdgt3_0,18). happensAt(walking(fomdgt3_0),18). orientation(fomdgt3_1,160,18). coords(fomdgt3_1,155,162,18).  visible(fomdgt3_1,18). happensAt(active(fomdgt3_1),18).
orientation(fomdgt3_0,179,19). coords(fomdgt3_0,239,216,19).  visible(fomdgt3_0,19). happensAt(walking(fomdgt3_0),19). orientation(fomdgt3_1,160,19). coords(fomdgt3_1,155,162,19).  visible(fomdgt3_1,19). happensAt(active(fomdgt3_1),19).
orientation(fomdgt3_0,179,20). coords(fomdgt3_0,240,219,20).  visible(fomdgt3_0,20). happensAt(walking(fomdgt3_0),20). orientation(fomdgt3_1,160,20). coords(fomdgt3_1,155,162,20).  visible(fomdgt3_1,20). happensAt(active(fomdgt3_1),20).
orientation(fomdgt3_0,179,21). coords(fomdgt3_0,241,222,21).  visible(fomdgt3_0,21). happensAt(walking(fomdgt3_0),21). orientation(fomdgt3_1,160,21). coords(fomdgt3_1,155,162,21).  visible(fomdgt3_1,21). happensAt(active(fomdgt3_1),21).
orientation(fomdgt3_0,179,22). coords(fomdgt3_0,242,225,22).  visible(fomdgt3_0,22). happensAt(walking(fomdgt3_0),22). orientation(fomdgt3_1,160,22). coords(fomdgt3_1,155,162,22).  visible(fomdgt3_1,22). happensAt(active(fomdgt3_1),22).
orientation(fomdgt3_0,179,23). coords(fomdgt3_0,243,230,23).  visible(fomdgt3_0,23). happensAt(walking(fomdgt3_0),23). orientation(fomdgt3_1,160,23). coords(fomdgt3_1,155,162,23).  visible(fomdgt3_1,23). happensAt(active(fomdgt3_1),23).
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
#show holdsAt/2.